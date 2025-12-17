# File: src/utils/serializer.ring
# Description: High-precision data serialization utility
# Author: Azzeddine Remmal

func SerializeData aData
    decimals(18)
    # Generate code string
    cCode = List2Code_Pretty(aData, 0)
    return cCode

# Recursive function with indentation support
func List2Code_Pretty aInput, nLevel
    # Indentation string (Tab or 4 spaces)
    cTab = copy(char(9), nLevel) 

    # Handle Numbers
    if isNumber(aInput) 
        return "" + aInput 
    ok

    # Handle Strings
    if isString(aInput)
        return '"' + substr(aInput, '"', '\"') + '"'
    ok

    # Handle Lists
    if isList(aInput)
        # Check if list is "Flat" (contains only numbers/strings, no sublists)
        isFlat = true
        nLen = len(aInput)
        for item in aInput
            if isList(item) or isObject(item) 
                isFlat = false 
                exit 
            ok
        next

        # Case 1: Flat List (Vector/Row) -> Write in ONE line
        if isFlat
            cOut = "["
            for i = 1 to nLen
                cOut += List2Code_Pretty(aInput[i], 0) # No indent needed inside line
                if i < nLen cOut += ", " ok
            next
            cOut += "]"
            return cOut
        
        # Case 2: Nested List (Matrix/Layers) -> Write with Newlines & Indent
        else
            cOut = "[" + nl
            for i = 1 to nLen
                # Add indentation for the item
                cOut += cTab + char(9) + List2Code_Pretty(aInput[i], nLevel + 1)
                
                if i < nLen 
                    cOut += "," + nl 
                else
                    cOut += nl # Last item gets newline before closing bracket
                ok
            next
            cOut += cTab + "]"
            return cOut
        ok
    ok
    
    return "NULL"


