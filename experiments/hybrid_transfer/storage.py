# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure both Python S3 readers and TensorStore's credential provider."""

import os

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret


def configure_storage():
    # TensorStore's C++ provider does not read Rigging's named secret files.
    os.environ["AWS_ACCESS_KEY_ID"] = require_secret("CW_KEY_ID")
    os.environ["AWS_SECRET_ACCESS_KEY"] = require_secret("CW_KEY_SECRET")
    configure_coreweave_s3()
