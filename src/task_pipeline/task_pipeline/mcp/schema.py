"""Generate exact nested ROS JSON schemas, without maintaining duplicate messages."""

import re

from rosidl_runtime_py.utilities import get_message


def field_schema(kind):
    """Describe generated ROS field types, including array and numeric bounds."""
    array = re.fullmatch(r'(.+)\[(\d*)\]', kind)
    sequence = re.fullmatch(r'(?:bounded_)?sequence<(.+?)(?:,\s*(\d+))?>', kind)
    if array or sequence:
        inner, limit = (array or sequence).groups()
        result = {'type': 'array', 'items': field_schema(inner)}
        if limit:
            result['maxItems'] = int(limit)
            if array:
                result['minItems'] = int(limit)
        return result
    if kind == 'boolean':
        return {'type': 'boolean'}
    if kind.startswith(('string', 'wstring')):
        result = {'type': 'string'}
        if '<=' in kind:
            result['maxLength'] = int(kind.split('<=')[1])
        return result
    match = re.fullmatch(r'(u?)int(8|16|32|64)', kind)
    if match:
        unsigned, bits = match.groups()
        bits = int(bits)
        return {'type': 'integer', 'minimum': 0 if unsigned else -(2**(bits-1)),
                'maximum': 2**bits-1 if unsigned else 2**(bits-1)-1}
    if kind in ('octet', 'char'):
        return {'type': 'integer', 'minimum': 0, 'maximum': 255}
    if kind in ('float', 'double', 'long double'):
        return {'type': 'number'}
    if '/' in kind:
        parts = kind.split('/')
        return message_schema(get_message('/'.join((parts[0], 'msg', parts[-1]))))
    raise ValueError('UNSUPPORTED_ROS_SCHEMA: '+kind)


def message_schema(message_type):
    """Require all actual ROS fields and reject unknown keys."""
    properties = {name: field_schema(kind) for name, kind in
                  message_type.get_fields_and_field_types().items()}
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}
