import json


class CompactListEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        # 自定义递归序列化函数
        def encode(obj, level=0):
            indent_str = ' ' * (self.indent * level if self.indent else 0)
            next_indent_str = ' ' * (self.indent * (level + 1) if self.indent else 0)

            if isinstance(obj, dict):
                items = []
                for k, v in obj.items():
                    items.append(f'\n{next_indent_str}{json.dumps(k, ensure_ascii=False)}: {encode(v, level + 1)}')
                return '{' + ','.join(items) + f'\n{indent_str}' + '}'

            elif isinstance(obj, list):
                # 列表保持一行
                inner = ', '.join(encode(v, level + 1) for v in obj)
                return f'[{inner}]'

            else:
                return json.dumps(obj, ensure_ascii=False)

        # yield 一次完整字符串（符合 JSONEncoder 接口要求）
        yield encode(o, 0)
