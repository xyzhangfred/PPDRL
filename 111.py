from collections import Counter

S = "ababcbacadefegdehijhklij"

class Solution:
    def partitionLabels(self, S: str):
        counts = Counter(S)
        sizes = []
        curr_letters = []
        curr_size = 0
        curr_count = 0

        for s in S:
            if s not in curr_letters:
                curr_letters.append(s)
                curr_count += counts[s]
            curr_count -= 1
            curr_size += 1
            if curr_count == 0:
                sizes.append(curr_size)
                curr_size = 0
                curr_letters = []
            print (curr_letters)
            print (curr_count)
        return sizes
sol = Solution()

sol.partitionLabels(S)
breakpoint()