#ifndef CTASKQUEUE
#define CTASKQUEUE

//轻量级的消息队列用于各种消息事件的通知
#include <list>
#include <mutex>
#include <condition_variable>

template<typename T>
class CTaskQueue{
public:
    CTaskQueue(int size = -1) :listSize(size > 0 ? size : INT_MAX) {
    }

    ~CTaskQueue(){}

    void Push(const T& element) {
        std::unique_lock<std::mutex> lock(blockMutex);
        //判断消息队列是否已满 等待处理
        fullNotify.wait(blobkMutex, [this]() {return this->blobkQueue.size() < this->listSize});
        blockQueue.push_front(element);
        emptyNotify.notify_all();   //唤醒所有的线程
    }

    T Pop() {
        std::unique_lock<std::mutex> lock(blockMutex);
        //如果队列为空，则等待获取消息
        emptyNotify.wait(blockMutex, [this]() {return !this->blockQueue.empty();});
        T ret = std::move(blockQueue.back());
        blockQueue.pop_back();
        fullNotify.notify_all();
        return std::move(ret);
    }

    void Clear(bool notify = true) {
        std::unique_lock<std::mutex> lock(blockMutex);
        while(!blockQueue.empty()) {
            //每次抛出一个消息就唤醒一个线程
            blockQueue.pop_front();
            fullNotify.notify_one();
        }
    }

    int Size() {
        std::unique_lock<std::mutex> lock(blockMutex);
        return blockQueue.size();
    }

private:
    int          listSize;    //消息队列的大小
    std::list<T> blockQueue;  //消息队列
    std::mutex   blockMutex;  //互斥锁
    std::condition_variable_any emptyNotify; // 队列为空的消息通知
    std::condition_variable_any fullNotify; //队列为满的消息通知
}
#endif // CTASKQUEUE

