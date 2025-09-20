# 链表（Linked List）讲义

## 1. 定义
链表是一种由节点（node）组成的线性数据结构。每个节点存储数据，并包含指向下一个节点的指针（或引用）。表头 head 只存储第一个有效节点的位置；最后一个节点的 next 指针为 nullptr。

## 2. 节点类（Node Class）
下面给出一个最小可用的节点类实现，包含读取数据、读取 next 指针以及设置 next 指针的方法。

```cpp
// C++
class Node {
private:
    int element;
    Node* next_node;

public:
    // 构造：默认值 element=0，next=nullptr
    Node(int e = 0, Node* n = nullptr) : element(e), next_node(n) {}

    // 访问器
    int retrieve() const { return element; }
    Node* next() const { return next_node; }

    // 修改 next 指针（用于删除/插入等操作）
    void set_next(Node* n) { next_node = n; }
};
```

## 3. 单链表类（List Class）
单链表类只需要保存指向第一个节点的指针 list_head。

```cpp
class List {
private:
    Node* list_head;

public:
    List() : list_head(nullptr) {}
    ~List() {
        // 释放所有节点，避免内存泄漏
        while (!empty()) {
            pop_front();
        }
    }
    bool empty() const {
        return list_head == nullptr;
    }
    // 在表头插入一个元素（头插）
    void push_front(int n) {
        list_head = new Node(n, list_head);
    }
    // 弹出表头元素（若为空则无操作）
    void pop_front() {
        if (empty()) return;
        Node* old_head = list_head;
        list_head = list_head->next();
        delete old_head;
    }
    // 读取表头元素
    int front() const {
        if (empty()) {
            throw std::runtime_error("List is empty");
        }
        return list_head->retrieve();
    }
    // 删除第一次出现的值 value，成功返回 true，未找到返回 false
    bool erase(int value) {
        Node* current = list_head;
        Node* previous = nullptr;

        // 找到第一个等于 value 的节点
        while (current != nullptr && current->retrieve() != value) {
            previous = current;
            current = current->next();
        }

        if (current == nullptr) return false;  // 未找到

        // 删除：分头结点与非头结点两种情况
        if (previous == nullptr) {
            // 删除头结点
            list_head = current->next();
        } else {
            previous->set_next(current->next());
        }

        delete current;
        return true;
    }
};
```

要点：
- push_front 始终是 O(1)。
- pop_front 若链表非空则 O(1)。
- erase 需要线性查找前驱节点，平均/最坏 O(n)。
- 为了能在 erase 中修改指针，Node 提供了 set_next；避免对私有成员的非法访问。

## 4. 单链表操作复杂度（假设可选 tail 指针用于 O(1) 的末端操作）
- 查找
  - 表头：Θ(1)
  - 第 k 个：O(n)
  - 表尾（有 tail 时）：Θ(1)
- 插入
  - 在某节点之后插入：Θ(1)（前提：已定位到该节点，定位本身 O(n)）
- 删除
  - 删除表头：Θ(1)
  - 删除第 k 个或表尾：O(n)（因为需要前驱）
- 查找前驱
  - 一般情况下 O(n)

关键弱点：需要“前驱”的操作（如在某节点前插入、按值删除）在单链表中往往退化为 O(n)。

## 5. 双向链表（Doubly Linked List）
双向链表在每个节点中增加指向前驱节点的指针 prev，从而能在 O(1) 时间访问前驱。

```cpp
template <typename T>
struct DoublyNode {
    T data;
    DoublyNode* next;
    DoublyNode* prev;

    DoublyNode(const T& v, DoublyNode* p = nullptr, DoublyNode* n = nullptr)
        : data(v), next(n), prev(p) {}
};
```

优势（在已定位到第 k 个节点的前提下）：
- 删除第 k 个：Θ(1)
- 在第 k 个之前插入：Θ(1)
- 查找前驱：Θ(1)

代价：多一个 prev 指针，空间额外为 Θ(n)。

## 6. 基于数组的节点存储（游标/光标实现）
也称“基于数组的链式存储”。用整型下标充当“指针”，-1 或常量 NULLIDX 表示空。

动机：
- 减少频繁 new/delete 带来的开销。
- 改善局部性与缓存友好性。

核心思想：
- 预先分配一个节点数组 nodes[]，每个节点包含 data 与 next 的下标。
- 维护“空闲链表”（free list），用 next 串起所有未使用的下标。
- 分配：从空闲链表头“弹出”一个下标。
- 释放：将下标“压回”空闲链表头。

示例：
```cpp
#include <stdexcept>
struct CursorNode {
    int data;
    int next; // -1 表示空
};
class CursorList {
    static const int CAP = 1024;      // 固定容量，可按需调整
    CursorNode nodes[CAP];            // 节点池
    int head;                         // 链表头（下标）
    int free_head;                    // 空闲链表头（下标）
public:
    CursorList() : head(-1), free_head(0) {
        // 初始化空闲链：0 -> 1 -> 2 -> ... -> CAP-1 -> -1
        for (int i = 0; i < CAP - 1; ++i) {
            nodes[i].next = i + 1;
        }
        nodes[CAP - 1].next = -1;
    }
    bool empty() const { return head == -1; }
    bool full()  const { return free_head == -1; }
    // 从空闲链分配一个下标；无可用空间返回 -1
    int allocate() {
        if (free_head == -1) return -1;
        int idx = free_head;
        free_head = nodes[idx].next;
        nodes[idx].next = -1;
        return idx;
    }
    // 释放一个下标，压回空闲链表头
    void deallocate(int idx) {
        nodes[idx].next = free_head;
        free_head = idx;
    }
    // 头插：成功返回 true；满了返回 false
    bool push_front(int value) {
        int idx = allocate();
        if (idx == -1) return false;     // 空间不足
        nodes[idx].data = value;
        nodes[idx].next = head;
        head = idx;
        return true;
    }

    // 弹出表头：空表返回 false
    bool pop_front() {
        if (empty()) return false;
        int old = head;
        head = nodes[head].next;
        deallocate(old);
        return true;
    }
    // 读取表头元素：空表抛异常
    int front() const {
        if (empty()) throw std::runtime_error("CursorList is empty");
        return nodes[head].data;
    }

    // 删除第一次出现的值 value：成功返回 true，未找到返回 false
    bool erase(int value) {
        int curr = head;
        int prev = -1;
        while (curr != -1 && nodes[curr].data != value) {
            prev = curr;
            curr = nodes[curr].next;
        }
        if (curr == -1) return false; // 未找到

        if (prev == -1) {
            // 删除头结点
            head = nodes[curr].next;
        } else {
            nodes[prev].next = nodes[curr].next;
        }
        deallocate(curr);
        return true;
    }

    // 清空链表（O(n)）
    void clear() {
        while (!empty()) pop_front();
    }
};
```

分析：
- 复杂度与普通链表一致（插入/删除/查找）。
- 只做一次大内存分配，减少内存碎片与系统调用。
- 容量固定，扩容需要整体重分配与拷贝，代价 O(CAP)。