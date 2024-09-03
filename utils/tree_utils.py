class TreeNode:
    """A simple tree node class."""
    def __init__(self, key, value, n_type, parent=None, children=None):
        self.key = key
        self.value = value
        self.n_type = n_type
        self.parent = parent
        self.children = children or []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class DFSBuilder:
    """A class to build a tree using depth-first approach."""
    def __init__(self, root_value):
        self.root = TreeNode(0, root_value, "variable")
        self.key_index = 0
        self.current_node = self.root
        self.stack = [self.root]
        self.visited = {self.root}

    def get_current_key(self):
        """Return the key of the current node."""
        return self.current_node.key

    def get_current_value(self):
        """Return the value of the current node."""
        return self.current_node.value

    def get_current_n_type(self):
        """Return the n_type of the current node."""
        return self.current_node.n_type

    def get_parent_value(self):
        """Return the value of the parent of the current node."""
        if len(self.stack) > 1:
            return self.stack[-2].value
        return None

    def add_child(self, value, n_type):
        """Add a new child to the current node."""
        # 检查当前节点的子节点中是否已存在相同value的节点
        for child in self.current_node.children:
            if child.value == value:
                return
        # 检查栈中是否已存在相同value的节点
        for node in self.stack:
            if node.value == value:
                return

        self.key_index += 1
        new_child = TreeNode(self.key_index, value, n_type)
        self.current_node.add_child(new_child)

    def move_to_parent(self):
        """Move back to the parent node."""
        if len(self.stack) > 1:
            self.stack.pop()  # Remove the current node
            self.current_node = self.stack[-1]  # Move to the parent
            return True
        else:
            return False

    def move_to_first_unvisited_child(self):
        """Move to the first unvisited child of the current node."""
        for child in self.current_node.children:
            if child not in self.visited:
                self.stack.append(child)
                self.current_node = child
                self.visited.add(child)  # Mark the child as visited
                return True
        return False

    def move_to_node_by_key(self, target_key):
        """Move to the node with the specified key by searching from the root."""
        def search_node(node, target_key):
            if node.key == target_key:
                return node
            for child in node.children:
                result = search_node(child, target_key)
                if result:
                    return result
            return None

        # 从根节点开始搜索
        result_node = search_node(self.root, target_key)
        if result_node:
            # 重建栈
            new_stack = []
            while result_node:
                new_stack.insert(0, result_node)
                result_node = result_node.parent

            self.stack = new_stack
            self.current_node = self.stack[-1]
            return True
        else:
            return False

    def delete_current_node(self):
        """Delete the current node and its subtree, then move to its parent."""
        if self.current_node == self.root:
            print("Cannot delete the root node.")
            return False

        parent_node = self.stack[-2]
        parent_node.children.remove(self.current_node)
        self.stack.pop()
        self.current_node = parent_node
        return True

    def continue_dfs(self):
        """Continue DFS from the current node."""
        if not self.stack:
            print("All nodes have been visited.")
            return False

        while self.stack:
            if not self.move_to_first_unvisited_child():
                if not self.move_to_parent():
                    return False
            else:
                break

        return True


def print_tree(node, level=0):
    print_str = "   " * level + "({0}) {1}".format(str(node.key), str(node.value))
    for child in node.children:
        print_str += "\n" + print_tree(child, level + 1)
    return print_str