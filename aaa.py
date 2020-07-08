class Solution:
    def combine(self, n: int, k: int) :
        def combine_helper(toSelect,prefix ):
            if len(prefix) == k: 
                res.append(prefix)
                return 
            
            for idx,val in enumerate(toSelect):
                combine_helper(toSelect[idx+1:],prefix + [val] )
        
        res = [] 
        combine_helper([i+1 for i in range(n)], [] )
        return res
sol = Solution()
res = sol.combine(5, 3)
print (res)