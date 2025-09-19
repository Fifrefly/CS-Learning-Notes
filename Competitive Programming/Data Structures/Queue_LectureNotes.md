## Queue
A queue is a linear data structure that follows the First In First Out (FIFO) principle. This means that the first element added to the queue will be the first one to be removed.
### Member Functions
template <typename Type>
void Queue<Type>::push(Type const &obj){
    if(queue_size==array_capacity){
        throw overflow;
    }
    ++iback;
    if(iback==array_capacity){
        iback=0;//循环操作
    }
    array[iback]=obj;
    ++queue_size;
}
template <typename Type>
Type Queue<Type>::pop(){
    if(empty()){
        throw underflow;
    }
    Type front = array[ifront];
    --queue_size;
    ++ifront;
    return front;
}
