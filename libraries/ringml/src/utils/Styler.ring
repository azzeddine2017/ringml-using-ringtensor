# File: Styler.ring
# Description: A helper class for styling console output with ANSI colors.
# Author: Azzeddine Remmal



# ============================================================================
# Console Styler Example
# ============================================================================

if isMainSourceFile() {
    oStyler = new Styler() {

		Info("Info")
		Success("Success" + "%")
		Warning("Warning")
		? Error("Error")

		# red color
		red(:BOLD, "Custom Usage")
		yellow(:UNDERLINE, "Custom Usage")
		green(:BOLD, "Custom Usage")
		blue(:UNDERLINE, "Custom Usage")
		magenta(:BOLD, "Custom Usage")
		cyan(:UNDERLINE, "Custom Usage")
		white(:BOLD, "Custom Usage")
		black(:UNDERLINE, "Custom Usage")	
		
		? seeString(:MAGENTA, :UNDERLINE, "Custom Usage")
	}
}

# ============================================================================
# Class: Styler
# Description: A helper class for styling console output with ANSI colors.
# ============================================================================

class Styler

	# ANSI Escape Code
	ESC = char(27)

	# Attributes for Colors and Styles
	aColors = []
	aStyles = []

	func init
		# Enable Unicode on Windows
		if isWindows()
			systemSilent("chcp 65001")
		ok
		setupDefinitions()
		return self
	# --------------------------------------------------------------------
	# Core Methods
	# --------------------------------------------------------------------

	# Returns the formatted string without printing it
	func costom(cColor, cStyle, cText)
		cColorCode = getColorCode(cColor)
		cStyleCode = getStyleCode(cStyle)
		
		cFullCode = cColorCode
		if cStyleCode != ""
			cFullCode = cStyleCode + ";" + cColorCode
		ok
		
		return ESC + "[" + cFullCode + "m" + cText + ESC + "[0m"

	# Prints the formatted string 
	func seeString(cColor, cStyle, cText)
		see costom(cColor, cStyle, cText)

	# --------------------------------------------------------------------
	# Helper Methods (Log Levels)
	# --------------------------------------------------------------------

	func Error(cMessage)
		see seeString(:BRIGHT_RED, :BOLD, "Error: " + cMessage)

	func Success(cMessage)
		see seeString(:BRIGHT_GREEN, :BOLD, "Success: " + cMessage)

	func Warning(cMessage)
		see seeString(:YELLOW, :NONE, "Warning: " + cMessage)

	func Info(cMessage)
		see seeString(:CYAN, :NONE, "Info: " + cMessage)

	# --------------------------------------------------------------------
	# Helper Methods (	Colors)
	# --------------------------------------------------------------------
	#  functions named colors
		

	func red(cStyle, cText)
		see seeString(:BRIGHT_RED, cStyle, cText)

	func green(cStyle, cText)
		see seeString(:BRIGHT_GREEN, cStyle, cText)

	func yellow(cStyle, cText)
		see seeString(:YELLOW, cStyle, cText)

	func cyan(cStyle, cText)
		see seeString(:CYAN, cStyle, cText)

	func magenta(cStyle, cText)
		see seeString(:MAGENTA, cStyle, cText)

	func blue(cStyle, cText)
		see seeString(:BLUE, cStyle, cText)

	func white(cStyle, cText)
		see seeString(:WHITE, cStyle, cText)

	func black(cStyle, cText)
		see seeString(:BLACK, cStyle, cText)
		
	func bright_blue(cStyle, cText)
		see seeString(:BRIGHT_BLUE, cStyle, cText)	
	
	func bright_magenta(cStyle, cText)
		see seeString(:BRIGHT_MAGENTA, cStyle, cText)	
	
	func bright_cyan(cStyle, cText)
		see seeString(:BRIGHT_CYAN, cStyle, cText)	
	
	func bright_white(cStyle, cText)
		see seeString(:BRIGHT_WHITE, cStyle, cText)		
	
	func bright_black(cStyle, cText)
		see seeString(:BRIGHT_BLACK, cStyle, cText)	

	func bright_yellow(cStyle, cText)
		see seeString(:BRIGHT_YELLOW, cStyle, cText)	

	func bright_red(cStyle, cText)
		see seeString(:BRIGHT_RED, cStyle, cText)		

	func bright_green(cStyle, cText)
		see seeString(:BRIGHT_GREEN, cStyle, cText)			

	# --------------------------------------------------------------------
	# Private / Internal Methods
	# --------------------------------------------------------------------
	
	private 

	func setupDefinitions
		aColors = [
			:RED = "31", :GREEN = "32", :YELLOW = "33", :BLUE = "34",
			:MAGENTA = "35", :CYAN = "36", :WHITE = "37", :BLACK = "30",
			:BRIGHT_RED = "1;31", :BRIGHT_GREEN = "1;32", :BRIGHT_YELLOW = "1;33",
			:BRIGHT_BLUE = "1;34", :BRIGHT_MAGENTA = "1;35", :BRIGHT_CYAN = "1;36",
			:BRIGHT_WHITE = "1;37"
		]

		aStyles = [
			:NONE = "", :BOLD = "1", :DIM = "2", 
			:UNDERLINE = "4", :BLINK = "5", 
			:REVERSE = "7", :HIDDEN = "8"
		]

	func getColorCode(vKey)
		if isString(vKey) 
			# If passed as string "RED", check if exists or return default
			if aColors[vKey] != NULL return aColors[vKey] ok
			return aColors[:WHITE]
		but isString(aColors[vKey])
			# If passed as list index/symbol :RED
			return aColors[vKey]
		ok
		return "37" # Default White

	func getStyleCode(vKey)
		if vKey = NULL return "" ok
		if isString(aStyles[vKey])
			return aStyles[vKey]
		ok
		return ""