import json
import sys
import argparse

def update_json_path(args):
    """
    Update configuration items in the JSON file
    """
    try:
        # read the json file
        with open(args.file_path, 'r') as f:
            data = json.load(f)
        
        # Iterate through the first configuration item in configurePresets array (assuming the target is the first one)
        configure_preset = data.get('configurePresets', [{}])[0]
        cache_variables = configure_preset.get('cacheVariables', {})


        # Modify the value of CMAKE_BUILD_TYPE
        if 'CMAKE_BUILD_TYPE' in cache_variables:
            cache_variables['CMAKE_BUILD_TYPE']['value'] = args.build_type
        else:
            print("CMAKE_BUILD_TYPE field not found")
            sys.exit(1)
        
        # Modify the value of ENABLE_SOURCE_PACKAGE
        if 'ENABLE_SOURCE_PACKAGE' in cache_variables:
            cache_variables['ENABLE_SOURCE_PACKAGE']['value'] = args.enable_source
        else:
            print("ENABLE_SOURCE_PACKAGE field not found")
            sys.exit(1)

        # Modify the value of vendor_name
        if 'vendor_name' in cache_variables:
            cache_variables['vendor_name']['value'] = args.vendor_name
        else:
            print("vendor_name field not found")
            sys.exit(1)

        # write back to JSON file (preserve indentation format)
        with open(args.file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print("Successfully updated parameters")

    except FileNotFoundError:
        print(f"File not found: {args.file_path}")
    except json.JSONDecodeError:
        print(f"JSON format error: {args.file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Modify configuration items in CMakePresets.json")
    parser.add_argument("file_path", help="Path to the JSON file")
    parser.add_argument("build_type", help="Build type (e.g., Debug or Release)")
    parser.add_argument("enable_source", help="Enable source package generation (true/false)")
    parser.add_argument("vendor_name", help="Specify the custom operator directory name")

    args = parser.parse_args()

    update_json_path(args)