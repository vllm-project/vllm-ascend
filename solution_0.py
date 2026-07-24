import logging
import os
import sys

# Import necessary libraries
from csrc.aclnn_torch_adapter.NPUBridge import NPUBridge

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_glm_version():
    # Check the version of the GLM5.1 library
    try:
        import glm
        version = glm.__version__
        logger.info(f"GLM5.1 library version: {version}")
        if version == "5.1.0":
            logger.warning("GLM5.1 library version is 5.1.0, which is known to have compatibility issues with PTA API.")
            return False
        else:
            return True
    except ImportError:
        logger.error("GLM5.1 library not found.")
        return False

def main():
    # Check the version of the GLM5.1 library
    if not check_glm_version():
        logger.error("GLM5.1 library version is not compatible with PTA API.")
        sys.exit(1)

    # Initialize the NPUBridge object
    try:
        bridge = NPUBridge()
        bridge.init()
        logger.info("NPUBridge initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize NPUBridge: {str(e)}")
        sys.exit(1)

    # Call the PTA API
    try:
        bridge.call_acl_api()
        logger.info("PTA API called successfully.")
    except Exception as e:
        logger.error(f"Failed to call PTA API: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()