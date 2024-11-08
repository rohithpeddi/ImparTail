from lib.stl.core.stl_formula import *


class IterativeParser:
    """Parses a list of tokens into an abstract syntax tree (AST) of STL Formulas iteratively.

    Additionally, it stores the identifiers in the formula from left to right.
    """

    def __init__(self):
        self.pos = 0
        self.tokens = None
        self.id_to_predicate_map = {}
        self.identifiers = []  # List to store identifiers in order

    def init_tokens(self, tokens, id_to_expression_map):
        self.tokens = tokens
        self.pos = 0
        self.id_to_predicate_map = id_to_expression_map
        self.identifiers = []  # Reset identifiers when initializing tokens

    def reset_pos(self):
        self.pos = 0
        self.tokens = None
        self.id_to_predicate_map = {}
        self.identifiers = []  # Reset identifiers when resetting

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

    def parse_formula(self):
        # Using a stack to manage operator precedence and associativity
        stack = []
        current_precedence = 0
        while True:
            token = self.current_token()
            if token[0] in ('EOF', 'RPAREN'):
                break

            if token[0] == 'ID':
                self.identifiers.append(token[1])  # Store identifier
                stack.append(self.id_to_predicate_map[token[1]])
                self.pos += 1
            elif token[0] in ('NEG', 'ALWAYS', 'EVENTUALLY'):
                self.pos += 1  # Skip the operator
                stack.append(self.parse_eventually())  # Handle unary operators recursively
            elif token[0] == 'LPAREN':
                self.pos += 1  # Skip the '('
                stack.append(self.parse_formula())  # Handle sub-formulas recursively
                self.expect('RPAREN')
            else:
                left = stack.pop() if stack else None
                self.pos += 1  # Skip the operator
                right = self.parse_primary()  # Find the next primary expression

                # Apply the operator
                if token[0] == 'AND':
                    left = And(left, right)
                elif token[0] == 'OR':
                    left = Or(left, right)
                elif token[0] == 'IMPL':
                    left = Implies(left, right)
                elif token[0] == 'UNTIL':
                    left = Until(left, right)
                elif token[0] == 'THEN':
                    left = Then(left, right)
                elif token[0] == 'EQ':
                    left = Equal(left, right)
                elif token[0] == 'LT':
                    left = LessThan(left, right)
                elif token[0] == 'GT':
                    left = GreaterThan(left, right)

                stack.append(left)

        return stack[0] if stack else None

    def parse_eventually(self):
        # Since 'eventually' and 'always' can nest, handle recursively
        token_type, _ = self.current_token()
        if token_type == 'ALWAYS':
            self.pos += 1
            return Always(self.parse_eventually())
        elif token_type == 'EVENTUALLY':
            self.pos += 1
            return Eventually(self.parse_eventually())
        else:
            return self.parse_negation()

    def parse_negation(self):
        if self.accept('NEG'):
            return Negation(self.parse_negation())
        else:
            return self.parse_primary()

    def parse_primary(self):
        if self.accept('LPAREN'):
            expr = self.parse_formula()
            self.expect('RPAREN')
            return expr
        elif self.accept('ID'):
            rel_name = self.tokens[self.pos - 1][1]
            self.identifiers.append(rel_name)
            return self.id_to_predicate_map[rel_name]
        else:
            raise RuntimeError(f'Unexpected token {self.current_token()}')

