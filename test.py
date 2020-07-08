# Definition for singly-linked list.
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def print_all(head):
    p = head
    while p is not None:
        print (p.val)
        p = p.next
    

class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        flag = 0
        new_head = ListNode(val = head.val, next = head.next)
        new_pointer = new_head
        if new_head.next is None:
            return new_head
        even_head = ListNode(val = head.next.val, next = head.next.next)
        even_pointer = even_head

        while even_pointer is not None:
            print ('even val: ', even_pointer.val)
            #flag marks if we have finished the first loop
            if even_pointer.next is None or even_pointer.next.next is None:
                even_pointer.next = None
                break
            else:
                even_pointer.next = even_pointer.next.next
            even_pointer = even_pointer.next
            
        print('even head')
        print_all(even_head)

        # print (head.val)
        while new_pointer is not None:
            print ('val: ', new_pointer.val)
            print ('flag: ', flag)

            #flag marks if we have finished the first loop
            if new_pointer.next is None or new_pointer.next.next is None:
                new_pointer.next = even_head
                return new_head
            else:
                new_pointer.next = new_pointer.next.next
         
            new_pointer = new_pointer.next
            
        return new_head



if __name__ == "__main__":
    test_head = ListNode(val=1, next=None)
    test_head.next = ListNode(val=2, next=None)
    test_head.next.next = ListNode(val=3, next=None)
    test_head.next.next.next = ListNode(val=4, next=None)
    test_head.next.next.next.next = ListNode(val=5, next=None)
    Sol = Solution()
    new_head = Sol.oddEvenList(test_head)
    print_all(new_head)
    breakpoint()