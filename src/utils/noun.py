def _singularize_noun(noun: str) -> str:
    """
    Convert plural noun to singular form using simple rules.
    
    Args:
        noun: Plural noun to convert
        
    Returns:
        Singular form of the noun
    """
    if not noun:
        return noun
    
    # Handle common irregular plurals
    irregular_plurals = {
        'children': 'child',
        'people': 'person',
        'men': 'man',
        'women': 'woman',
        'feet': 'foot',
        'teeth': 'tooth',
        'mice': 'mouse',
        'geese': 'goose'
    }
    
    if noun in irregular_plurals:
        return irregular_plurals[noun]
    
    # Handle regular plural patterns
    if noun.endswith('ies') and len(noun) > 3:
        # companies -> company, stories -> story
        return noun[:-3] + 'y'
    elif noun.endswith('ves') and len(noun) > 3:
        # knives -> knife, wolves -> wolf
        return noun[:-3] + 'f'
    elif noun.endswith('ses') and len(noun) > 3:
        # glasses -> glass, classes -> class
        return noun[:-2]
    elif noun.endswith('es') and len(noun) > 2:
        # boxes -> box, dishes -> dish
        if noun.endswith(('ches', 'shes', 'xes', 'zes')):
            return noun[:-2]
        else:
            return noun[:-1]
    elif noun.endswith('s') and len(noun) > 1:
        # books -> book, cars -> car
        return noun[:-1]
    
    return noun

def _pluralize_noun(noun: str) -> str:
    """
    Convert singular noun to plural form using simple rules.
    
    Args:
        noun: Singular noun to convert
        
    Returns:
        Plural form of the noun
    """
    if not noun:
        return noun
    
    # Handle common irregular singulars
    irregular_singulars = {
        'child': 'children',
        'person': 'people',
        'man': 'men',
        'woman': 'women',
        'foot': 'feet',
        'tooth': 'teeth',
        'mouse': 'mice',
        'goose': 'geese'
    }

    if noun in irregular_singulars:
        return irregular_singulars[noun]

    # Handle regular plural patterns
    if noun.endswith('y') and len(noun) > 2:
        # baby -> babies, city -> cities
        return noun[:-1] + 'ies'
    elif noun.endswith('f') and len(noun) > 2:
        # knife -> knives, wolf -> wolves
        return noun[:-1] + 'ves'
    elif noun.endswith('s') and len(noun) > 2:
        # glass -> glasses, class -> classes
        return noun
    elif noun.endswith('o') and len(noun) > 2:
        # photo -> photos, piano -> pianos
        return noun + 's'
    elif len(noun) > 1:
        # book -> books, car -> cars
        return noun + 's'

    return noun
