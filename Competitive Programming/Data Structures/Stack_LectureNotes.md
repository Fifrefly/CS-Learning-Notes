# 栈（Stack）

栈是一种遵循后进先出（LIFO, Last In First Out）原则的线性数据结构：最后入栈的元素最先出栈。

## 数组实现（Array Implementation）

### Top
```cpp
template <typename Type>
Type Stack<Type>::top() const {
    if(empty()){
        throw underflow;
    }
    return array[stack_size-1];
}
```

### Pop
```cpp
template <typename Type>
Type Stack<Type>::pop(){
    if(empty()){
        throw underflow;
    }
    --stack_size;
    return array[stack_size];
}
```

### Push
```cpp
template <typename Type>
void Stack<Type>::push(Type const &obj){
    if(stack_size==array_capacity){
        throw overflow;
    }
    array[stack_size]=obj;
    ++stack_size;
}
```

## 链表实现（Linked List Implementation）

### Top
```cpp
template <typename Type>
Type Stack<Type>::top() const {
    if(empty()){
        throw underflow;
    }
    return head->data;
}
```

### Pop
```cpp
template <typename Type>
Type Stack<Type>::pop(){
    if(empty()){
        throw underflow;
    }
    Node *old=head;
    Type obj=head->data;
    head=head->next;
    delete old;
    --stack_size;
    return obj;
}
```

### Push
```cpp
template <typename Type>
void Stack<Type>::push(Type const &obj){
    Node *newNode=new Node;
    newNode->data=obj;
    newNode->next=head;
    head=newNode;
    ++stack_size;
}
```

## 扩容策略与摊还时间复杂度（Array Capacity & Amortized Time）

摊还时间（Amortized time）定义：若 n 次操作总耗时为 Θ(f(n))，则每次操作的摊还时间为 Θ(f(n)/n)。

当数组实现的栈空间耗尽时，需要扩容；扩容策略对性能影响显著。

- 情况一：按常数增量扩容（例如每次 +1）
  - 总拷贝次数：0 + 1 + 2 + ... + (n - 1) = Θ(n²)
  - 摊还时间：Θ(n²) / n = Θ(n)，低效

- 情况二：按倍率扩容（例如容量翻倍）
  - 总拷贝次数：在 1, 2, 4, 8, ..., 2^k < n 处扩容，总拷贝约为 2n - 1 = Θ(n)
  - 摊还时间：Θ(n) / n = Θ(1)，高效