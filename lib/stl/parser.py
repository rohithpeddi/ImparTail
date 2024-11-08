from lib.stl.core.stl_formula import Negation, GreaterThan, LessThan, Equal, Until, Then, And, Or, Implies
from lib.stl.core.stl_formula import Always, Eventually


class Parser:
    """Parses a list of tokens into an abstract syntax tree (AST) of STL Formulas."""

    def __init__(self):
        self.pos = 0
        self.tokens = None
        self.id_to_predicate_map = {}

    def init_tokens(self, tokens, id_to_expression_map):
        self.tokens = tokens
        self.pos = 0
        self.id_to_predicate_map = id_to_expression_map

    def reset_pos(self):
        self.pos = 0
        self.tokens = None
        self.id_to_predicate_map = {}

    def current_token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        else:
            return ('EOF', '')

    def accept(self, kind):
        if self.current_token()[0] == kind:
            self.pos += 1
            return True
        return False

    def expect(self, kind):
        if not self.accept(kind):
            raise RuntimeError(f'Expected {kind}, got {self.current_token()[0]}')

    # Parsing functions with precedence climbing
    def parse_formula(self):
        return self.parse_implication()

    def parse_implication(self):
        left = self.parse_or()
        while self.accept('IMPL'):
            right = self.parse_or()
            left = Implies(left, right)
        return left

    def parse_or(self):
        left = self.parse_and()
        while self.accept('OR'):
            right = self.parse_and()
            left = Or(left, right)
        return left

    def parse_and(self):
        left = self.parse_until()
        while self.accept('AND'):
            right = self.parse_until()
            left = And(left, right)
        return left

    def parse_until(self):
        left = self.parse_then()
        while self.accept('UNTIL'):
            right = self.parse_then()
            left = Until(left, right)
        return left

    def parse_then(self):
        left = self.parse_equality()
        while self.accept('THEN'):
            right = self.parse_equality()
            left = Then(left, right)
        return left

    def parse_equality(self):
        left = self.parse_relational()
        while self.accept('EQ'):
            right = self.parse_relational()
            left = Equal(left, right)
        return left

    def parse_relational(self):
        left = self.parse_eventually()
        while True:
            if self.accept('LT'):
                right = self.parse_eventually()
                left = LessThan(left, right)
            elif self.accept('GT'):
                right = self.parse_eventually()
                left = GreaterThan(left, right)
            else:
                break
        return left

    def parse_eventually(self):
        token_type, _ = self.current_token()
        if self.accept('ALWAYS'):
            subformula = self.parse_eventually()
            return Always(subformula)
        elif self.accept('EVENTUALLY'):
            subformula = self.parse_eventually()
            return Eventually(subformula)
        else:
            return self.parse_negation()

    def parse_negation(self):
        if self.accept('NEG'):
            subformula = self.parse_negation()
            return Negation(subformula)
        else:
            return self.parse_primary()

    def parse_primary(self):
        if self.accept('LPAREN'):
            expr = self.parse_formula()
            self.expect('RPAREN')
            return expr
        elif self.accept('ID'):
            rel_name = self.tokens[self.pos - 1][1]
            return self.id_to_predicate_map[rel_name]
        else:
            raise RuntimeError(f'Unexpected token {self.current_token()}')