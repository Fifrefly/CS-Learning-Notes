## Linked List
### Definition
A linked list is a data structure where each object is stored in a node. As well as storing data, the node must also contain a reference/pointer to the node containing the next item of data.
Ps. head(表头)只存储了下一个节点的位置，最后一个node只存储了值和nullptr
### Node Class
Class Node{
    private:
        int element;
        Node *next_node;
    public:
        Node(int = 0,Node * = nullptr);
        int retrieve() const;
        Node * next() const;
};
#### Node Constructor
Node::Node( int e, Node *n ):element( e ),next_node( n ) {
}
#### Accessors
int Node::retrieve() const {
    return element;
}
Node *Node::next() const {
    return next_node;
}
### Linked List Class
Because each node in a linked lists refers to the next, the linked list class need only link to the first node in the list
The linked list class requires member variable:  a pointer to a node

class List {
    private:
        Node *list_head;
    // ...
};

#### void push_front(int)
void List::push_front(int n){
    if (empty()) {
        list_head = new Node(n,nullptr);
    } else {
        list_head = new Node(n,list_head);
    }
}
#### void pop_front(int &)
void List::pop_front(){
    if (!empty()) {
        Node *old_head = list_head;
        list_head = list_head->next();
        delete old_head;
    }
}
#### bool empty() const
bool List::empty() const {
    return list_head == nullptr;
}
#### int front() const
int List::front() const {
    if (!empty()) {
        return list_head->retrieve();
    }
    throw "List is empty";
}
#### Node* get_head() const
Node* List::get_head() const {
    return list_head;
}
#### void erase(int)

### Double Linked List

### Node-based storage with arrays