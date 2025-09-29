import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from http import HTTPStatus
import os
from vllm.entrypoints.openai.protocol import ErrorResponse


class BaseValidator(ABC):
    def __init__(self,
                 param_name: str,
                 min_val:Union[float, int, None] = None,
                 max_val:Union[float, int, None] = None,
                 type_: Optional[Type] = None,
                 error_msg: Optional[str] = None,
                 subfield: Optional[list[str]] = None
                 ):
        self.param_name = param_name
        self.error_msg = error_msg or f"{param_name} is not supported."
        self.min_val = min_val
        self.max_val = max_val
        self.type_ = type_
        self.subfield = subfield

        @abstractmethod
        def validate(self, value: Any) -> Optional[str]:
            pass


class SupportedValidator(BaseValidator):
    def check_subfield_dict(self, value):
        for param_name, _ in value.items():
            if param_name not in self.subfield:
                return f"{self.param_name}:{param_name} is not supported."
        return None
    
    def check_subfield_list(self, value):
        for val in value:
            if isinstance(val, Dict):
                for param_name, _ in val.items():
                    if param_name not in self.subfield:
                        return f"{self.param_name}:{param_name} is not supported."
        return None

    def validate(self, value: Any) -> Optional[str]:
        # The value must be included within the subfield.
        if self.subfield is not None:
            if isinstance(value, Dict):
                return self.check_subfield_dict(value)
            if isinstance(value, list):
                return self.check_subfield_list(value)
        return None
    

class UnsupportedValidator(BaseValidator):
    def validate(self, value: Any) -> Optional[str]:
        return self.error_msg


class RangeValidator(BaseValidator):
    def validate(self, value: Any) -> Optional[str]:
        if not isinstance(value, self.type_):
            return f"{self.param_name} must be of type {self.type_.__name__}"
        if not (self.min_val <= value <= self.max_val):
            return f"{self.param_name} must between {self.min_val} and {self.max_val}, got {value}."
        return None


def load_validators_from_json(config_path: str) -> Dict[str, BaseValidator]:
    validators: Dict[str, BaseValidator] = {}
    if config_path == "":
        return validators
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    type_map = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool
    }

    for param_name, spec in config.items():
        vtype = spec["type"]

        if vtype == "supported":
            validators[param_name] = SupportedValidator(
                param_name,
                subfield=spec.get("subfield")
            )
        
        elif vtype == "validated":
            dtype_str = spec["data_type"]
            dtype = type_map.get(dtype_str)

            validators[param_name] = RangeValidator(
                param_name,
                data_type=dtype,
                min_val=spec.get("min_val"),
                max_val=spec.get("max_val"),
                subfield=spec.get("subfield")
            )
        elif vtype == "unsupported":
            validators[param_name] = UnsupportedValidator(
                param_name
            )
        else:
            raise ValueError(f"validators config json ValueError, Unknown validator type: {vtype} for {param_name}")
    
    return validators


VALIDATORS: Dict[str, BaseValidator] = load_validators_from_json(os.getenv("VALIDATORS_CONFIG_PATH", ""))


class ValidateSamplingParams(BaseHTTPMiddleware):
    def create_error_response(self, status_code, error):
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                message=str(error),
                type="BadRequestError",
                code=status_code.value
            ).model_dump()
        )
    
    def replace_with_stars(self, text):
        return "*" * len(text)
    
    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in ("/v1/completions", "/v1/chat/completions"):
            body = await request.body()
            if not body:
                return await call_next(request)
            
            try:
                json_load = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                return await call_next(request)
            
            if json_load.get("kv_transfer_params"):
                max_tokens = json_load.get("max_tokens", -1)
                if max_tokens < 0:
                    json_load["max_tokens"] = int(os.getenv("DEFAULT_MAX_TOKENS", 8192))
                    request._body = json.dumps(json_load).encode("utf-8")
                return await call_next(request)
            
            if not VALIDATORS:
                return await call_next(request)
            
            status_code = HTTPStatus.BAD_REQUEST

            for param_name, value in json_load.items():
                validator = VALIDATORS.get(param_name)
                if not validator:
                    return self.create_error_response(status_code, f"{param_name} is not supported.")
                error = validator.validate(value)
                if error is not None:
                    return self.create_error_response(status_code, error)
                
        if request.method == "GET" and request.url.path == "/v1/models":
            response = await call_next(request)
            chunk = await anext(response.body_iterator)
            chunk_json = json.loads(chunk.decode("utf-8"))
            
            if chunk_json is not None and len(chunk_json.get("data", [])) > 0 and chunk_json.get("data")[0].get("root"):
                chunk_json.get("data")[0]["root"] = self.replace_with_stars(chunk_json.get("data")[0].get("root"))
            
            new_json_str = json.dumps(chunk_json, ensure_ascii=False)
            new_chunk = new_json_str.encode("utf-8")

            return Response(
                content=new_chunk,
                headers={
                    "Content-Length": str(len(new_chunk)),
                    'content-type': 'application/json'
                }
            )

        return await call_next(request)