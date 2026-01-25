# File: functions.ring
# Description: Helper functions for RingML.
# Author: Azzeddine Remmal



# ============================================================================
# Function: hasAttribute
# Description: Checks if an object has a specific attribute.	
# ============================================================================
func hasAttribute oObj, cName
        aAttrs = attributes(oObj)
        for a in aAttrs
            if lower(a) = lower(cName) return true ok
        next
        return false
# ============================================================================
# Function: Randomize
# Description: Initializes the random number generator with a seed.
# ============================================================================
func Randomize(nSeed)
    return Random(nSeed)

# ============================================================================
# Function: listToTensor
# Description: Converts a list to a Tensor.
# ============================================================================
func listToTensor aList
    nRows = len(aList)
    if nRows = 0 return new Tensor(1,1) ok
    nCols = len(aList[1])
    
    oTen = new Tensor(nRows, nCols)
    
    for r = 1 to nRows
        for c = 1 to nCols
            oTen.setVal(r, c, aList[r][c])
        next
    next
    return oTen

# ============================================================================
# Function: Sci2Dec
# Description: Converts a scientific notation string to a decimal number.
# ============================================================================  
func Sci2Dec cNum
    cNum = lower(trim(string(cNum)))
    
    # Check if 'e' exists
    if substr(cNum, "e"){  
        return 0 + cNum
    else 
        raise("Invalid scientific notation: " + cNum ) 
    }
    
    # Split Base and Exponent
    aParts = split(cNum, "e")
    if len(aParts) < 2 {
        raise("Invalid scientific notation: " + cNum ) 
    }
    
    nBase = 0 + aParts[1]
    nExp  = 0 + aParts[2]
    
    # Calculate: Base * 10^Exp
    return nBase * pow(10, nExp)

# ============================================================================
# Function: round
# Description: Rounds a number to a specified number of decimal places.
# ============================================================================  
func round nNum, nDecimals
    return floor(nNum * pow(10, nDecimals)) / pow(10, nDecimals)    

# ============================================================================
# Function: pad
# Description: Pads a string with spaces to a specified length.
# ============================================================================
func pad x, n
    cStr = "" + x
    if len(cStr) > n cStr = left(cStr, n) ok
    return cStr + copy(" ", n - len(cStr))

# ============================================================================
# Function: formatTime
# Description: Formats a time duration in seconds.
# ============================================================================
func formatTime nSec
    nSec = floor(nSec)
    nH = floor(nSec / 3600)
    nM = floor((nSec % 3600) / 60)
    nS = nSec % 60
    
    cStr = ""
    if nH > 0 cStr += "" + nH + "h " ok
    if nM > 0 cStr += "" + nM + "m " ok
    cStr += "" + nS + "s"
    return cStr

#===================================================================================
# Function: SerializeData
# Description: Serializes a data structure into a code string.
#===================================================================================
func SerializeData aData
    decimals(18)
    # Generate code string
    cCode = List2Code_Pretty(aData, 0)
    return cCode

#===================================================================================
# Function: List2Code_Pretty
# Description: Recursive function with indentation support
#===================================================================================
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
    
#===================================================================================
# Function: flatList
# Description: Flattens a nested list into a single list.
#===================================================================================
func flatList aList
    aOutput = []
    for item in aList
        if isList(item)
            for i in item
                aOutput + i
            next
        else
            aOutput + item
        ok
    next
    return aOutput
#===================================================================================
# Function: calc_norm
# Description: Calculates the norm of a list of pointers.
#===================================================================================
func calcListNorm aList
    sum_sq = 0
    for item in aList
        # Unpacking: The item is a list containing the indicator
        if isList(item)
            ptr = item[1]
        else
            ptr = item
        ok
        
        sum_sq += tensor_sum_squares(ptr)
    next
    return sqrt(sum_sq)
    
#===================================================================================
# Function: calc_single_norm
# Description: Calculates the norm of a single pointer.
#===================================================================================
func calcSingleNorm pPtr
    return sqrt(tensor_sum_squares(pPtr))
    