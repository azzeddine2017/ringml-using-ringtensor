# File: src/utils/cleaner.ring
# Description: Clean RST Text
# Author: Azzeddine Remmal

func CleanRST cText
    aLines = str2list(cText)
    aClean = []
    
    for cLine in aLines
        cTrimmed = trim(cLine)
        
        # RST Directives Removal
        if left(cTrimmed, 2) = ".." loop ok
        if left(cTrimmed, 3) = "===" loop ok
        if left(cTrimmed, 3) = "---" loop ok
        
        # Remove empty lines if you want compact text
        # if len(cTrimmed) = 0 loop ok
        
        aClean + cLine 
    next
    
    return list2str(aClean)