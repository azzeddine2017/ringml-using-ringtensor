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



