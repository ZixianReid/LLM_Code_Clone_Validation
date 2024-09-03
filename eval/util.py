import difflib
def line_based_similarity(row):
    # Code1 and code2 are extracted from the row
    code1 = row['func1']
    code2 = row['func2']

    # Split each code into lines
    lines1 = code1.strip().split('\n')
    lines2 = code2.strip().split('\n')

    # Create a Differ object
    differ = difflib.Differ()

    # Calculate differences
    diff = list(differ.compare(lines1, lines2))

    # Calculate similarity
    matcher = difflib.SequenceMatcher(None, lines1, lines2)
    return matcher.ratio()

