import keyword
import re
from pathlib import Path
import marshal
import importlib.util
import sys
import threading
import logging

# -------------------------------
# Настройка логирования
# -------------------------------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# -------------------------------
# Токенизация
# -------------------------------
def tokenize_source_code(source: str) -> list[tuple[str, str]]:
    log.debug(f"Начата токенизация исходного кода: {source!r}")
    token_spec = [
        ("NUMBER", r"\d+(\.\d+)?"),
        ("ID", r"[A-Za-z_]\w*"),
        ("STRING", r"(\".*?\"|\'.*?\')"),
        ("NEWLINE", r"\n"),
        ("SKIP", r"[ \t]+"),
        ("OP", r"[\+\-\*/%=<>!]+"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COLON", r":"),
        ("COMMA", r","),
        ("MISMATCH", r"."),
    ]

    tok_regex = "|".join(f"(?P<{name}>{pattern})" for name, pattern in token_spec)
    tokens: list[tuple[str, str]] = []

    for idx, mo in enumerate(re.finditer(tok_regex, source), start=1):
        kind = mo.lastgroup
        value = mo.group()
        log.debug(f"[Токенизация] Итерация {idx}: найден токен ({kind}, {value})")
        match kind:
            case "NUMBER":
                tokens.append(("NUMBER", value))
            case "ID":
                if value in keyword.kwlist:
                    tokens.append(("KEYWORD", value))
                else:
                    tokens.append(("ID", value))
            case "STRING":
                tokens.append(("STRING", value))
            case "NEWLINE" | "SKIP":
                log.debug(f"[Токенизация] Пропускаем символ: {value!r}")
                continue
            case "OP" | "LPAREN" | "RPAREN" | "COLON" | "COMMA":
                tokens.append((kind, value))
            case "MISMATCH":
                raise SyntaxError(f"Непредвиденный символ: {value}")
    log.debug(f"Все токены: {tokens}")
    return tokens

# -------------------------------
# AST и исполнение
# -------------------------------
def build_ast_from_tokens(tokens: list[tuple[str, str]]) -> tuple:
    log.debug(f"Начато построение AST из токенов: {tokens}")
    pos = 0
    stack_nodes = []
    stack_ops = []

    while pos < len(tokens):
        tok = tokens[pos]
        log.debug(f"[AST] Итерация позиции {pos}: токен {tok}")

        if tok[0] == "NUMBER":
            value = float(tok[1]) if '.' in tok[1] else int(tok[1])
            log.debug(f"[AST] Добавляем число: {value}")
            stack_nodes.append(("num", value))
            pos += 1
        elif tok[0] == "ID":
            log.debug(f"[AST] Добавляем переменную: {tok[1]}")
            stack_nodes.append(("var", tok[1]))
            pos += 1
        elif tok[0] == "OP" and tok[1] in ("+", "-", "*", "/"):
            op = tok[1]
            log.debug(f"[AST] Обработка оператора: {op}")
            while stack_ops:
                top = stack_ops[-1]
                log.debug(f"[AST] Сравнение приоритета с верхом стека: {top}")
                if (op in ("+", "-") and top in ("+", "-", "*", "/")) or (op in ("*", "/") and top in ("*", "/")):
                    right = stack_nodes.pop()
                    left = stack_nodes.pop()
                    applied_op = stack_ops.pop()
                    log.debug(f"[AST] Создаём бинарную операцию: ({left} {applied_op} {right})")
                    stack_nodes.append(("binop", applied_op, left, right))
                else:
                    break
            stack_ops.append(op)
            pos += 1
        elif tok[0] == "LPAREN":
            log.debug(f"[AST] Начало скобочной группы на позиции {pos}")
            depth = 1
            sub_pos = pos + 1
            while sub_pos < len(tokens) and depth > 0:
                t = tokens[sub_pos]
                log.debug(f"[AST] Скобочная итерация {sub_pos}: {t}, глубина {depth}")
                if t[0] == "LPAREN":
                    depth += 1
                elif t[0] == "RPAREN":
                    depth -= 1
                sub_pos += 1
            if depth != 0:
                raise SyntaxError("Несбалансированная '('")
            sub_ast = build_ast_from_tokens(tokens[pos + 1:sub_pos - 1])
            stack_nodes.append(sub_ast)
            log.debug(f"[AST] Добавлен AST из скобок: {sub_ast}")
            pos = sub_pos
        elif tok[0] == "RPAREN":
            raise SyntaxError("Непредвиденная ')'")
        else:
            raise SyntaxError(f"Непредвиденный токен: {tok}")

    while stack_ops:
        right = stack_nodes.pop()
        left = stack_nodes.pop()
        applied_op = stack_ops.pop()
        log.debug(f"[AST] Формируем бинарную операцию после всех токенов: ({left} {applied_op} {right})")
        stack_nodes.append(("binop", applied_op, left, right))

    if len(stack_nodes) != 1:
        raise SyntaxError("Некорректное выражение")

    log.debug(f"Построенный AST: {stack_nodes[0]}")
    return stack_nodes[0]

def compile_ast_to_bytecode(ast: tuple):
    log.debug(f"Компиляция AST в байткод: {ast}")

    def ast_to_expr(node):
        match node[0]:
            case "num":
                return str(node[1])
            case "var":
                return node[1]
            case "binop":
                op, left, right = node[1], node[2], node[3]
                return f"({ast_to_expr(left)} {op} {ast_to_expr(right)})"
            case _:
                raise ValueError(f"Неизвестный AST узел: {node[0]}")

    expr_str = ast_to_expr(ast)
    log.debug(f"Python-выражение для компиляции: {expr_str}")
    return compile(expr_str, "<ast>", "eval")

def run_bytecode(code_obj, globals_ns=None):
    log.debug(f"Выполнение байткода с namespace: {globals_ns}")
    if globals_ns is None:
        globals_ns = {}
    result = eval(code_obj, globals_ns)
    log.debug(f"Результат выполнения: {result}")
    return result

# -------------------------------
# Интерпретатор с присваиваниями
# -------------------------------
GIL = threading.Lock()

def gil_execute(func, *args, **kwargs):
    with GIL:
        return func(*args, **kwargs)

def execute_source(source_code: str):
    log.info("Начало исполнения исходного кода")
    lines = source_code.strip().splitlines()
    namespace = {}
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        log.debug(f"[Цикл строк] Итерация {line_no}: {line}")
        if not line:
            continue
        if "=" in line:
            var_name, expr = line.split("=", 1)
            var_name = var_name.strip()
            expr = expr.strip()
            log.debug(f"[Присваивание] {var_name} = {expr}")
            expr_tokens = tokenize_source_code(expr)
            ast = build_ast_from_tokens(expr_tokens)
            code_obj = compile_ast_to_bytecode(ast)
            namespace[var_name] = eval(code_obj, {}, namespace)
            log.info(f"[Результат присваивания] {var_name} = {namespace[var_name]}")
        else:
            expr_tokens = tokenize_source_code(line)
            ast = build_ast_from_tokens(expr_tokens)
            code_obj = compile_ast_to_bytecode(ast)
            result = eval(code_obj, {}, namespace)
            log.info(f"[Результат выражения] {line} = {result}")
    log.info(f"Окончательный namespace: {namespace}")
    return namespace

# -------------------------------
# Пример
# -------------------------------
source_code = """
x = 5
y = 10
z = x * 2 + y
"""

namespace = execute_source(source_code)
print("Namespace after execution:", namespace)
