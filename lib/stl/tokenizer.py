import re


class Tokenizer:
    """Converts input strings into a list of tokens."""

    # Define token specifications as tuples of (TOKEN_NAME, REGEX_PATTERN)
    token_specification = [
        ('ALWAYS',       r'G'),                         # Always operator
        ('NEG',     r'¬'),                              # Negation operator
        ('AND',     r'∧'),                              # Conjunction operator
        ('OR',      r'∨'),                              # Disjunction operator
        ('IMPL',    r'⇒'),                              # Implication operator
        ('EVENTUALLY', r'F'),                           # Eventually operator
        ('UNTIL',   r'U'),                              # Until operator
        ('LT',r'<='),                              # Less than operator
        ('GT',r'>='),                           # Greater than operator
        ('EQ',r'=='),                           # Equal to operator
        ('THEN',    r'T'),                             # Then operator
        ('LPAREN',  r'\('),                             # Left Parenthesis
        ('RPAREN',  r'\)'),                             # Right Parenthesis
        ('ID',      r'[a-zA-Z_][a-zA-Z0-9_]*'),         # Identifiers
        ('SKIP',    r'[ \t]+'),                         # Skip spaces and tabs
        ('MISMATCH',r'.'),                              # Any other character
    ]

    def __init__(self):
        # Compile the regular expression for tokenizing
        self.token_regex = re.compile('|'.join(f'(?P<{name}>{pattern})'
                                              for name, pattern in self.token_specification))

    def tokenize(self, code):
        """Tokenizes the input string into a list of (TOKEN_NAME, VALUE) tuples."""
        tokens = []
        for mo in self.token_regex.finditer(code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'SKIP':
                continue
            elif kind == 'MISMATCH':
                raise RuntimeError(f'Unexpected character {value!r}')
            else:
                tokens.append((kind, value))
        return tokens