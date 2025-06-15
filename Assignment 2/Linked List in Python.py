class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:
    def __init__(self):
        self.start = None

    def append(self, value):
        new_node = Node(value)
        if self.start is None:
            self.start = new_node
            return
        pointer = self.start
        while pointer.next:
            pointer = pointer.next
        pointer.next = new_node

    def display(self):
        if self.start is None:
            print("The list is empty.")
            return
        temp = self.start
        while temp:
            print(temp.value, end=" -> ")
            temp = temp.next
        print("null")

    def remove_at_position(self, pos):
        if self.start is None:
            print("Cannot delete from an empty list.")
            return

        if pos <= 0:
            print("Invalid position. Position must be 1 or greater.")
            return

        if pos == 1:
            print(f"Deleted node with value {self.start.value} at position {pos}")
            self.start = self.start.next
            return

        prev = None
        current = self.start
        index = 1

        while current and index < pos:
            prev = current
            current = current.next
            index += 1

        if current is None:
            print(f"Position {pos} exceeds the length of the list.")
            return

        print(f"Deleted node with value {current.value} at position {pos}")
        prev.next = current.next


# Sample usage
if __name__ == "__main__":
    lst = LinkedList()
    lst.append(5)
    lst.append(15)
    lst.append(25)
    lst.append(35)

    print("Original list:")
    lst.display()

    lst.remove_at_position(3)
    print("After deleting 3rd node:")
    lst.display()

    lst.remove_at_position(10)  # out-of-range
    lst.remove_at_position(0)   # invalid index
