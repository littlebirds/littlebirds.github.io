class Solution {
public:
    bool isMatch(string s, string p) { 
        const char* pStr = s.c_str();
        const char* pPat = p.c_str(); 
        return isMatch(pStr, pPat);        
    }
   
    bool isMatch(const char* pStr, const char* pPattern) {
        if ('\0' == *pStr && '\0' == *pPattern) { 
            return true;
        }  
        if ('\0' == *pPattern) {
            return false;
        } 
        if (pPattern[1] == '*') {
            // match 0 occurance
            if (isMatch(pStr, pPattern +2)) {
                return true;
            }
            while (pPattern[0] == '.' && *pStr != '\0' || pStr[0] == pPattern[0]) {
                ++pStr;
                if (isMatch(pStr, pPattern+2)) {
                    return true;
                } 
            }
            return false;
        } else {
            if (pStr[0] == pPattern[0] || pPattern[0] == '.' && pStr[0] != '\0') {
                return isMatch(pStr+1, pPattern+1);
            } else {
                return false;
            }
        }       
    }    
};
