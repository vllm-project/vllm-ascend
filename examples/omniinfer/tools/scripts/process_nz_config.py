import json
import sys

config_file = ""
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    print("file name is empty")
    exit()

with open(config_file, 'r')as f:
    config = json.load(f)

size = len(config["Reshape"]["input0"]["dtype"].split(','))

if size == 26:
    config["Reshape"]["input0"]["dtype"] = "int4," + config["Reshape"]["input0"]["dtype"]
    config["Reshape"]["input0"]["format"] = "FRACTAL_NZ," + config["Reshape"]["input0"]["format"]
    config["Reshape"]["input0"]["unknownshape_format"] = "FRACTAL_NZ," + config["Reshape"]["input0"]["unknownshape_format"]

    config["Reshape"]["input1"]["dtype"] = "int32," + config["Reshape"]["input1"]["dtype"]
    config["Reshape"]["input1"]["format"] = "ND," + config["Reshape"]["input1"]["format"]
    config["Reshape"]["input1"]["unknownshape_format"] = "ND," + config["Reshape"]["input1"]["unknownshape_format"]

    config["Reshape"]["output0"]["dtype"] = "int4," + config["Reshape"]["output0"]["dtype"]
    config["Reshape"]["output0"]["format"] = "FRACTAL_NZ," + config["Reshape"]["output0"]["format"]
    config["Reshape"]["output0"]["unknownshape_format"] = "FRACTAL_NZ," + config["Reshape"]["output0"]["unknownshape_format"]

    format_list = config["GroupedMatmul"]["input1"]["format"].split(',')
    format_list[-6] = "FRACTAL_NZ"
    new_format = ','.join(format_list)
    config["GroupedMatmul"]["input1"]["format"] = new_format

    bit_cast = '{"attr":{"list":"boxes"},"attr_boxes":{"defaultValue":"3","paramType":"optional","type":"int","value":"all"},"dynamicRankSupport":{"flag":"true"},"dynamicShapeSupport":{"flag":"true"},"input0":{"dtype":"int32","format":"FRACTAL_NZ","name":"x","paramType":"required","shape":"all"},"opFile":{"value":"Null"},"output0":{"dtype":"int4","format":"FRACTAL_NZ","name":"y","paramType":"required","shape":"all"}}'
    bit_cast = json.loads(bit_cast)
    config["Bitcast"] = bit_cast

    if '910b' in config_file:
        gmmfr_format_list = config["GroupedMatmulFinalizeRouting"]["input1"]["format"].split(',')
        gmmfr_format_list[2] = "FRACTAL_NZ"
        new_format = ','.join(gmmfr_format_list)
        config["GroupedMatmulFinalizeRouting"]["input1"]["format"] = new_format
        config["GroupedMatmulFinalizeRouting"]["input1"]["unknownshape_format"] = new_format
        config["Bitcast"]["heavyOp"] = {"flag": "true"}

    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    print("process aic-ascend910_93-ops-info.json done")
else:
    print("nothing changed")
