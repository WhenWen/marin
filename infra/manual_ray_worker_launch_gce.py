#!/usr/bin/env python3
# This script manually spins up a GCE VM and connects it to the provided Ray cluster head node.
# Usage: python infra/manual_ray_worker_launch_gce.py --cluster_yaml <cluster_yaml> --head <head-node-ip>\
#          --machine_type <machine_type> [--project <project> --zone <zone> --vm_name <vm_name> --image <image>\
#          --scopes <scopes>]
import argparse
import subprocess
import tempfile
import time

import levanter.infra.cli_helpers as cli
import yaml


def run_command(*cmd, print_command=True, check=True):
    """
    Helper function to run a shell command, optionally printing it,
    and optionally checking for nonzero exit code.
    """
    if print_command:
        print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=check)
    return result


def gcloud_compute_create_vm(
    vm_name,
    machine_type,
    project,
    zone,
    image,
    scopes=None,
):
    """
    Create a GCE VM using gcloud CLI.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "create",
        vm_name,
        f"--machine-type={machine_type}",
        f"--project={project}",
        f"--zone={zone}",
        f"--image={image}",
    ]

    # handle scopes
    # e.g. if the user wants "cloud-platform" or "storage-rw", etc.
    if scopes:
        cmd.append(f"--scopes={scopes}")

    run_command(*cmd)


def gcloud_compute_ssh(vm_name, zone, command_list=None, project=None):
    """
    SSH into the GCE VM and run the given command(s).
    command_list can be a list of commands, or a single string.
    """
    if command_list is None:
        command_list = []
    if isinstance(command_list, str):
        # single command
        command_list = [command_list]

    for cmd in command_list:
        run_command("gcloud", "compute", "ssh", vm_name, "--zone", zone, "--project", project, "--command", cmd)


def setup_vm_docker_gce(
    vm_name,
    zone,
    project=None,
):
    """
    Installs Docker on the GCE VM if needed.
    This is a minimal example using apt on Debian/Ubuntu-based images.
    Adjust for your OS as needed.
    """
    # Update and install Docker
    commands = ["sudo apt-get update -y", "sudo apt-get install -y docker.io", "sudo usermod -aG docker $USER"]
    gcloud_compute_ssh(vm_name, zone, commands, project=project)


def main():
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(parser, config, ["--cluster_yaml"], required=True)
    cli.add_arg(parser, config, ["--head"], required=True, help="address of head node")
    cli.add_arg(parser, config, ["--machine_type"], required=True, help="GCE machine type, e.g. n1-standard-4")

    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--vm_name"], required=False, default=None)
    cli.add_arg(parser, config, ["--image"], default="ubuntu-2004-focal-v20250111", help="Name of GCE disk image")
    cli.add_arg(parser, config, ["--zone"], default=None, type=str, required=False, help="GCP zone for the VM")
    cli.add_arg(parser, config, ["--scopes"], default="cloud-platform", help="Scopes for the instance")

    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    project = args.project
    zone = args.zone
    image = args.image
    machine_type = args.machine_type

    if zone is None:
        zone = cli.gcloud_config()["zone"]
    if zone is None:
        raise ValueError("Zone must be specified or set in gcloud config.")

    vm_name = args.vm_name
    if vm_name is None:
        vm_name = f"ray-worker-manual-{cli.default_run_id()}"

    head = args.head

    # Load cluster YAML
    with open(args.cluster_yaml, "r") as f:
        cluster_yaml = yaml.safe_load(f)

    # Docker info from cluster_yaml
    image_id = cluster_yaml["docker"]["image"]
    container_name = cluster_yaml["docker"].get("container_name", "ray")
    worker_run_options = cluster_yaml["docker"].get("worker_run_options", ["-v", "/tmp:/tmp"])

    # Setup commands inside container or host
    initialization_commands = cluster_yaml.get("initialization_commands", [])
    setup_commands = cluster_yaml.get("setup_commands", [])

    entry_command = f"ray start --address={head}:6379 --block"

    # Create the GCE VM
    print(f"Creating GCE VM {vm_name} in project {project}, zone {zone}")
    gcloud_compute_create_vm(
        vm_name=vm_name,
        machine_type=machine_type,
        project=project,
        zone=zone,
        image=image,
        scopes=args.scopes,
    )

    # Wait a few seconds for SSH to become available
    time.sleep(15)

    # Set up Docker on the new VM
    print(f"Setting up Docker on VM {vm_name} ...")
    setup_vm_docker_gce(vm_name, zone, project=project)

    # Remove any existing container with the same name
    print(f"Removing any existing Docker container named {container_name} ...")
    gcloud_compute_ssh(vm_name, zone, f"docker rm -f {container_name} || true", project=project)

    # Create an entrypoint script that runs the setup commands and the ray start
    print(f"Creating and copying entry script to VM {vm_name} ...")
    with tempfile.NamedTemporaryFile("w", prefix="entry", suffix=".sh", delete=False) as tf:
        tf.write("#!/usr/bin/env bash\n\n")
        # Container-level setup commands
        for cmd in setup_commands:
            tf.write(cmd + "\n")
        # The final command that blocks the container while in the Ray cluster
        tf.write(entry_command + "\n")

        tf.flush()
        local_entry = tf.name

    # Copy the entry script to the VM
    run_command(
        "gcloud", "compute", "scp", local_entry, f"{vm_name}:/tmp/entry.sh", "--zone", zone, "--project", project
    )

    # Make the script executable
    gcloud_compute_ssh(vm_name, zone, "chmod a+rx /tmp/entry.sh", project=project)

    # Run initialization commands on the VM (host-level)
    for cmd in initialization_commands:
        gcloud_compute_ssh(vm_name, zone, cmd, project=project)

    # Build the Docker run command
    docker_command = [
        "docker",
        "run",
        "-d",
        "--net=host",
        f"--name={container_name}",
        "--init",
        "--privileged",
        *worker_run_options,
        image_id,
        "/bin/bash",
        "/tmp/entry.sh",
    ]

    print(f"Starting container with command: {docker_command}")

    # SSH to the VM and launch the container
    gcloud_compute_ssh(vm_name, zone, " ".join(docker_command), project=project)

    print(
        f"Done! The VM {vm_name} should now be running a Docker container "
        f"named '{container_name}' connected to the Ray head at {head}:6379."
    )


if __name__ == "__main__":
    main()
