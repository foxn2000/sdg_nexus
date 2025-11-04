"""
MABEL Expression Language (MEX) - v2 式エンジン
"""
from __future__ import annotations
import re, random, json
from typing import Any, Dict, List, Optional
from datetime import datetime

class MEXEvaluator:
    """MEX式を評価するエンジン"""
    
    def __init__(self, context: Dict[str, Any], globals_vars: Optional[Dict[str, Any]] = None):
        self.context = context  # ローカル変数（出力値など）
        self.globals_vars = globals_vars or {}  # グローバル変数
        self.call_stack_depth = 0
        self.max_depth = 256
    
    def eval(self, expr: Any) -> Any:
        """式を評価"""
        if expr is None:
            return None
        if isinstance(expr, (str, int, float, bool)):
            return expr
        if isinstance(expr, list):
            return [self.eval(e) for e in expr]
        if not isinstance(expr, dict):
            return expr
        
        # 単一キーの演算子式
        if len(expr) == 1:
            op, args = next(iter(expr.items()))
            return self._eval_op(op, args)
        
        # 複数キー（オブジェクトリテラル）
        return {k: self.eval(v) for k, v in expr.items()}
    
    def _eval_op(self, op: str, args: Any) -> Any:
        """演算子を評価"""
        # 論理演算
        if op == "and":
            return all(self._truthy(self.eval(a)) for a in args)
        if op == "or":
            return any(self._truthy(self.eval(a)) for a in args)
        if op == "not":
            return not self._truthy(self.eval(args))
        
        # 比較演算
        if op == "eq":
            a, b = args
            return self.eval(a) == self.eval(b)
        if op == "ne":
            a, b = args
            return self.eval(a) != self.eval(b)
        if op == "lt":
            a, b = args
            return self._to_num(self.eval(a)) < self._to_num(self.eval(b))
        if op == "le":
            a, b = args
            return self._to_num(self.eval(a)) <= self._to_num(self.eval(b))
        if op == "gt":
            a, b = args
            return self._to_num(self.eval(a)) > self._to_num(self.eval(b))
        if op == "ge":
            a, b = args
            return self._to_num(self.eval(a)) >= self._to_num(self.eval(b))
        
        # 算術演算
        if op == "add":
            return sum(self._to_num(self.eval(a)) for a in args)
        if op == "sub":
            vals = [self._to_num(self.eval(a)) for a in args]
            return vals[0] - sum(vals[1:]) if len(vals) > 1 else -vals[0]
        if op == "mul":
            result = 1
            for a in args:
                result *= self._to_num(self.eval(a))
            return result
        if op == "div":
            a, b = args
            return self._to_num(self.eval(a)) / self._to_num(self.eval(b))
        if op == "mod":
            a, b = args
            return self._to_num(self.eval(a)) % self._to_num(self.eval(b))
        if op == "pow":
            a, b = args
            return self._to_num(self.eval(a)) ** self._to_num(self.eval(b))
        if op == "neg":
            return -self._to_num(self.eval(args))
        
        # 文字列演算
        if op == "concat":
            return "".join(str(self.eval(a)) for a in args)
        if op == "split":
            text, sep = args
            return str(self.eval(text)).split(str(self.eval(sep)))
        if op == "replace":
            text, old, new = args
            return str(self.eval(text)).replace(str(self.eval(old)), str(self.eval(new)))
        if op == "lower":
            return str(self.eval(args)).lower()
        if op == "upper":
            return str(self.eval(args)).upper()
        if op == "trim":
            return str(self.eval(args)).strip()
        if op == "len":
            val = self.eval(args)
            return len(val) if hasattr(val, '__len__') else 0
        
        # コレクション演算
        if op == "map":
            lst, fn = args["list"], args["fn"]
            items = self.eval(lst)
            if not isinstance(items, list):
                items = [items]
            return [self.eval(fn) for item in items]
        if op == "filter":
            lst, fn = args["list"], args["fn"]
            items = self.eval(lst)
            if not isinstance(items, list):
                items = [items]
            return [item for item in items if self._truthy(self.eval(fn))]
        if op == "any":
            return any(self._truthy(self.eval(a)) for a in args)
        if op == "all":
            return all(self._truthy(self.eval(a)) for a in args)
        if op == "unique":
            lst = self.eval(args)
            if isinstance(lst, list):
                seen = set()
                result = []
                for item in lst:
                    key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item
                    if key not in seen:
                        seen.add(key)
                        result.append(item)
                return result
            return lst
        if op == "sort":
            lst = self.eval(args)
            return sorted(lst) if isinstance(lst, list) else lst
        if op == "slice":
            lst, start = args["list"], args.get("start", 0)
            end = args.get("end")
            items = self.eval(lst)
            return items[self.eval(start):self.eval(end) if end is not None else None]
        
        # 正規表現
        if op == "regex_match":
            text, pattern = args["text"], args["pattern"]
            return bool(re.search(str(self.eval(pattern)), str(self.eval(text))))
        if op == "regex_extract":
            text, pattern = args["text"], args["pattern"]
            matches = re.findall(str(self.eval(pattern)), str(self.eval(text)))
            return matches
        if op == "regex_replace":
            text, pattern, repl = args["text"], args["pattern"], args["replacement"]
            return re.sub(str(self.eval(pattern)), str(self.eval(repl)), str(self.eval(text)))
        
        # 制御構造
        if op == "if":
            cond = args["cond"]
            then_val = args["then"]
            else_val = args.get("else")
            return self.eval(then_val) if self._truthy(self.eval(cond)) else self.eval(else_val)
        if op == "case":
            for when_clause in args.get("when", []):
                cond = when_clause.get("cond")
                then_val = when_clause.get("then")
                if self._truthy(self.eval(cond)):
                    return self.eval(then_val)
            return self.eval(args.get("else"))
        
        # 変数参照
        if op == "var":
            name = str(args)
            # グローバル変数を優先、次にコンテキスト
            if name in self.globals_vars:
                return self.globals_vars[name]
            return self.context.get(name)
        if op == "ref":
            # 出力名参照（コンテキストのみ）
            return self.context.get(str(args))
        if op == "get":
            obj = self.eval(args.get("obj") or args[0] if isinstance(args, list) else args)
            path = str(self.eval(args.get("path") or args[1] if isinstance(args, list) else ""))
            default = self.eval(args.get("default")) if isinstance(args, dict) else None
            return self._get_path(obj, path, default)
        
        # 代入
        if op == "set":
            var_name = str(args["var"])
            value = self.eval(args["value"])
            self.globals_vars[var_name] = value
            return value
        
        # 時間・乱数
        if op == "now":
            return datetime.now().isoformat()
        if op == "rand":
            low = self.eval(args.get("min", 0))
            high = self.eval(args.get("max", 1))
            return random.uniform(low, high)
        
        # 型変換
        if op == "to_number":
            try:
                return float(self.eval(args))
            except:
                return 0
        if op == "to_string":
            return str(self.eval(args))
        if op == "to_boolean":
            return self._truthy(self.eval(args))
        if op == "parse_json":
            return json.loads(str(self.eval(args)))
        if op == "stringify":
            return json.dumps(self.eval(args), ensure_ascii=False)
        
        # 未知の演算子
        raise ValueError(f"Unknown MEX operator: {op}")
    
    def _truthy(self, v: Any) -> bool:
        """真偽値判定"""
        if v is None or v is False:
            return False
        if v == "" or v == 0 or v == [] or v == {}:
            return False
        return True
    
    def _to_num(self, v: Any) -> float:
        """数値変換"""
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except:
                return 0.0
        return 0.0
    
    def _get_path(self, obj: Any, path: str, default: Any = None) -> Any:
        """パス指定で値を取得（a.b[0].c 形式）"""
        if not path:
            return obj
        
        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]
        
        current = obj
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part, default)
            elif isinstance(current, list):
                try:
                    idx = int(part)
                    current = current[idx] if 0 <= idx < len(current) else default
                except:
                    current = default
            else:
                current = default
            
            if current is None:
                return default
        
        return current if current is not None else default


def eval_mex(expr: Any, context: Dict[str, Any], globals_vars: Optional[Dict[str, Any]] = None) -> Any:
    """MEX式を評価するヘルパー関数"""
    evaluator = MEXEvaluator(context, globals_vars)
    return evaluator.eval(expr)
