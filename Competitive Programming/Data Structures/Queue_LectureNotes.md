## 队列（Queue）
队列是一种线性数据结构，遵循 FIFO（First In First Out，先进先出）原则：最先进入的元素最先被取出。
## 循环数组实现（Circular Array）
思路：使用固定容量的数组作为环，并通过取模运算实现“首尾相接”的效果。维护三个状态：head、tail、size。

- head：当前队首元素的下标
- tail：下一次写入的位置（即“队尾的后一个位置”）
- size：队列中的元素数量

好处：避免频繁移动元素；空间已满时可选择扩容并做“归一化拷贝”。

```cpp
#include <stdexcept>

template <typename T>
class CircularQueue {
private:
    T* array;
    int head;
    int tail;
    int queue_size;
    int array_capacity;

    void resize_if_needed() {
        if (queue_size < array_capacity) return;
        // 扩容至 2 倍，并按逻辑顺序从 0 开始拷贝
        int new_capacity = array_capacity == 0 ? 1 : array_capacity * 2;
        T* new_array = new T[new_capacity];
        for (int i = 0; i < queue_size; ++i) {
            new_array[i] = array[(head + i) % array_capacity];
        }
        delete[] array;
        array = new_array;
        array_capacity = new_capacity;
        head = 0;
        tail = queue_size;
    }

public:
    CircularQueue(int capacity = 8)
        : array(new T[capacity]), head(0), tail(0), queue_size(0), array_capacity(capacity) {}

    ~CircularQueue() { delete[] array; }

    bool empty() const { return queue_size == 0; }
    int size() const { return queue_size; }

    void push(T const& obj) {
        resize_if_needed();
        array[tail] = obj;
        tail = (tail + 1) % array_capacity;
        ++queue_size;
    }

    T pop() {
        if (queue_size == 0) throw std::underflow_error("Queue is empty");
        T front_val = array[head];
        head = (head + 1) % array_capacity;
        --queue_size;
        return front_val;
    }

    T& front() {
        if (queue_size == 0) throw std::underflow_error("Queue is empty");
        return array[head];
    }

    T const& front() const {
        if (queue_size == 0) throw std::underflow_error("Queue is empty");
        return array[head];
    }

    T& back() {
        if (queue_size == 0) throw std::underflow_error("Queue is empty");
        int last_idx = (tail - 1 + array_capacity) % array_capacity;
        return array[last_idx];
    }

    T const& back() const {
        if (queue_size == 0) throw std::underflow_error("Queue is empty");
        int last_idx = (tail - 1 + array_capacity) % array_capacity;
        return array[last_idx];
    }
};
```

要点与边界：
- tail 指向“下一个可写入位置”，因此最后一个元素下标是 (tail - 1 + capacity) % capacity。
- 扩容时必须“归一化”（从 head 起按逻辑顺序拷贝），否则环形断裂导致顺序错误。

---

## 链表实现（Linked List）

思路：维护 head（队首）与 tail（队尾）两个指针，保证 push 与 pop 均为 Θ(1)。

```cpp
#include <stdexcept>

template <typename T>
struct Node {
    T data;
    Node* next;
    Node(T const& d, Node* n=nullptr) : data(d), next(n) {}
};

template <typename T>
class LinkedListQueue {
private:
    Node<T>* head; // 队首
    Node<T>* tail; // 队尾
    int queue_size;

public:
    LinkedListQueue() : head(nullptr), tail(nullptr), queue_size(0) {}
    ~LinkedListQueue() {
        while (head) {
            Node<T>* tmp = head;
            head = head->next;
            delete tmp;
        }
    }

    bool empty() const { return queue_size == 0; }
    int size() const { return queue_size; }

    void push(T const& obj) {
        Node<T>* node = new Node<T>(obj);
        if (!head) {
            head = tail = node;
        } else {
            tail->next = node;
            tail = node;
        }
        ++queue_size;
    }

    T pop() {
        if (!head) throw std::underflow_error("Queue is empty");
        Node<T>* tmp = head;
        T val = tmp->data;
        head = head->next;
        if (!head) tail = nullptr; // 移除最后一个元素后需同步置空 tail
        delete tmp;
        --queue_size;
        return val;
    }

    T& front() {
        if (!head) throw std::underflow_error("Queue is empty");
        return head->data;
    }

    T const& front() const {
        if (!head) throw std::underflow_error("Queue is empty");
        return head->data;
    }

    T& back() {
        if (!tail) throw std::underflow_error("Queue is empty");
        return tail->data;
    }

    T const& back() const {
        if (!tail) throw std::underflow_error("Queue is empty");
        return tail->data;
    }
};
```

要点与边界：
- 空队列 push 时需要同时设置 head 与 tail。
- pop 后若 head 为空，必须将 tail 也置空，避免“悬空尾指针”。
- 链表无需扩容；注意析构释放所有节点，防止内存泄漏。

---

## 复杂度与实现对比

- 循环数组：
    - push/pop/front/back/empty/size：均为 Θ(1) 均摊（扩容触发时单次为 O(n)）
    - 空间是连续的；适合频繁访问和缓存友好
    - 需处理扩容与元素拷贝
- 链表：
    - push/pop/front/back/empty/size：均为 Θ(1)
    - 不需要扩容；插入/删除稳定
    - 额外指针开销，不够缓存友好

---

## STL `std::queue` 速查与常见用法

头文件与类型：
- 头文件：<queue>
- 默认容器：`std::deque<T>`（也可指定 `std::list<T>` 或 `std::vector<T>`，但 vector 不适合频繁头删）

常用接口：
- `q.push(x)`, `q.pop()`（无返回值）
- `q.front()`, `q.back()`
- `q.empty()`, `q.size()`

示例：
```cpp
#include <queue>

std::queue<int> q;
q.push(1);
q.push(2);
int a = q.front(); // 1
q.pop();           // 移除 1
int b = q.front(); // 2
```

---
