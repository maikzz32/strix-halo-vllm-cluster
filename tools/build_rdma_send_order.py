"""Generate cyclic peer posting for the fixed all-to-all transport; arithmetic unchanged."""


def generate(source):
    start = source.index('static int collective(')
    end = source.index('\nstatic void copy_error(', start)
    body = source[start:end]
    original = 'for (int r = 0; r < NR; ++r) if (r != s->rank) {'
    assert body.count(original) == 1
    replacement = 'for (int step = 1; step < NR; ++step) {\n        int r = (s->rank + step) % NR;'
    return source[:start] + body.replace(original, replacement) + source[end:]
