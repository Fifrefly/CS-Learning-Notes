## Stack
A stack is a linear data structure that follows the Last In First Out (LIFO) principle. This means that the last element added to the stack will be the first one to be removed.
### Array Implementation
#### Top
template <typename Type>
Type Stack<Type>::top() const {
    if(empty()){
        throw underflow;
    }
    return array[stack_size-1];
}
#### Pop
template <typename Type>
Type Stack<Type>::pop(){
    if(empty()){
        throw underflow;
    }
    --stack_size;
    return array[stack_size];
}
#### Push
template <typename Type>
void Stack<Type>::push(Type const &obj){
    if(stack_size==array_capacity){
        throw overflow;
    }
    array[stack_size]=obj;
    ++stack_size;
}
### Array Capacity
To state the average run time, the amortized time is:
if n operations requires Θ(f(n)) time, then the amortized time per operation is Θ(f(n)/n).
Therefore, if inserting n objects requires:
Θ(n^2) time, then the amortized time per operation is Θ(n).
Θ(n) time, then the amortized time per operation is Θ(1).
Case1: If the array is full,we increase the capacity by 1 each time.
Suppose we insert k objects
- The pushing of the kth object requires k-1 copies.
Case2: If the array is full, we double the capacity.
