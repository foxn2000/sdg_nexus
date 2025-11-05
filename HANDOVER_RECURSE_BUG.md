# 再帰関数（op: recurse）実装バグの引き継ぎ

## 現在の状況

MABEL v2の`op: recurse`（再帰関数）機能の実装に問題があり、常に0を返す不具合が発生しています。

## 問題の詳細

### テストケース
`examples/test_recurse_simple.yaml`で5の階乗（5! = 120）を計算するテストを実行すると、期待値120ではなく0.0が返されます。

### デバッグ出力（DEBUG_RECURSE=1で実行）
```
[DEBUG] recurse depth=4, args={'n': 1.0}
[DEBUG] base case reached, returning {'result': 1}
[DEBUG] recursive call returned: {'result': 1}
[DEBUG] executing step: {'op': 'set', 'var': 'result', 'value': {'mul': [{'var': 'n'}, {'var': 'prev'}]}}
[DEBUG] step result: {'result': 0.0}  ← ★問題箇所
```

### 問題の核心

1. **再帰呼び出しの戻り値は正しい**: `{'result': 1}`が正常に返されている
2. **戻り値の格納も成功**: `prev`という変数名で`call_ctx`に格納されている（デバッグログ確認済み）
3. **MEX評価で値が失われる**: `{'mul': [{'var': 'n'}, {'var': 'prev'}]}`を評価すると`prev`が0になる

### 原因の推定

`sdg/executors.py`の`_execute_logic_step`関数内：
```python
def _execute_logic_step(step: Dict[str, Any], ctx: Dict[str, Any], exec_ctx: ExecutionContext) -> Dict[str, Any]:
    op = step.get("op")
    
    if op == "set":
        var_name = step.get("var", "result")
        # MEX評価時にローカルコンテキストをマージ
        eval_ctx = {**exec_ctx.globals_vars, **ctx}
        value = eval_mex(step.get("value"), ctx, eval_ctx)  # ★ここで問題発生
        ...
```

**問題点**: 
- `eval_mex`の第2引数（context）に`ctx`を渡しているが、MEXエンジン（`sdg/mex.py`）の`var`演算子が正しくこのコンテキストから値を取得できていない可能性
- または、`eval_ctx`の構築方法が間違っている

### 関連ファイル

1. **sdg/executors.py**: 
   - `_execute_logic_step`関数（約540行目）
   - `_apply_logic_block`内の`op: recurse`実装（約270-400行目）

2. **sdg/mex.py**:
   - `MEXEvaluator.eval`メソッド
   - `_eval_op`メソッド内の`var`演算子処理（約130行目付近）

3. **テストファイル**:
   - `examples/test_recurse_simple.yaml`: シンプルな階乗テスト
   - `examples/data/test_all_input.jsonl`: テスト入力データ

## 修正すべき箇所

### 優先度1: MEX評価のコンテキスト渡し

`sdg/mex.py`の`MEXEvaluator`を確認し、`var`演算子が正しくコンテキストを参照しているか検証：

```python
# 現在の実装（mex.py）
if op == "var":
    name = str(args)
    # グローバル変数を優先、次にコンテキスト
    if name in self.globals_vars:
        return self.globals_vars[name]
    return self.context.get(name)  # ★ここで正しく取得できているか？
```

**修正案**:
- `eval_mex`呼び出し時のコンテキスト構築を見直す
- MEXエンジンの初期化時に正しくローカル変数が含まれるようにする

### 優先度2: デバッグ情報の追加

問題特定のため、`_execute_logic_step`内にデバッグログを追加：
```python
if os.environ.get("DEBUG_RECURSE") == "1":
    print(f"[DEBUG] MEX eval context keys: {list(ctx.keys())}", file=sys.stderr)
    print(f"[DEBUG] eval_ctx keys: {list(eval_ctx.keys())}", file=sys.stderr)
    print(f"[DEBUG] prev value in ctx: {ctx.get('prev')}", file=sys.stderr)
```

## 動作確認方法

```bash
# デバッグモードで実行
DEBUG_RECURSE=1 python -m sdg run \
  --yaml examples/test_recurse_simple.yaml \
  --input examples/data/test_all_input.jsonl \
  --output examples/output_recurse.jsonl

# 結果確認
cat examples/output_recurse.jsonl
# 期待値: {"result": "120"}
# 現在値: {"result": "0.0"}
```

## 既知の動作する機能

以下の機能は正常に動作しています：
- ✅ `op: set` (変数代入)
- ✅ `op: while` (反復処理)
- ✅ `op: reduce` (リスト畳み込み)
- ✅ `op: call` (ユーザ定義関数呼び出し)
- ✅ `op: let` (ローカル束縛)
- ✅ MEXの基本演算（`add`, `mul`, `sub`など）
- ✅ MEXの`var`演算子（グローバル変数参照時）

## 回避策

現時点では、再帰処理が必要な場合はPythonブロックを使用：

```yaml
- type: python
  function_code: |
    def factorial(n):
        if n <= 1: return 1
        return n * factorial(n-1)
    
    def main(ctx, n: int) -> dict:
        return {"Result": factorial(n)}
  entrypoint: main
  inputs: {n: 5}
  outputs: [Result]
```

## 次のステップ

1. MEXエンジンの`var`演算子がローカルコンテキストから正しく値を取得できるか検証
2. `eval_mex`の引数の渡し方を修正
3. 必要に応じて、再帰呼び出し時のコンテキスト管理方法を見直す
4. テストケースで動作確認
5. 他の機能（while, reduceなど）に影響がないことを確認

## 参考情報

- MABEL v2仕様書: `mabel_v2.md`（§14.5に再帰関数の例題あり）
- 実装状況: `mabel_v2.md`の§16に記載
- 全機能テスト: `examples/test_all_features.yaml`
