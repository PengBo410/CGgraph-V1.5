#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Thread/bindCore.hpp"
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace CPJ {

class ThreadPool
{
  public:
    ThreadPool() = delete;
    ThreadPool(size_t);
    ThreadPool(size_t, std::vector<int>&); /* Bind to core*/
    template <class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    bool isEmpty();
    void clearPool();
    ~ThreadPool();

  private:
    // need to keep track of threads so we can join them
    std::vector<std::thread> workers;
    // the task queue
    std::queue<std::function<void()>> tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) : stop(false)
{
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
            for (;;)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock,
                                         [this]
                                         {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
}

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads, std::vector<int>& coreId_vec) : stop(false)
{
    assert_msg(threads == coreId_vec.size(), "threads can not match coreId_vec.size()");
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this, coreId_vec, i]
            {
            bindCore(coreId_vec[i]);
            // Msg_info("ThreadPool[%zu] bind to logic core[%d]", i, sched_getcpu());
            for (;;)
            {
                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock,
                                         [this]
                                         {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }

                task();
            }
        });
}

// add new work item to the pool
template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace(
            [task]()
            {
            (*task)();
        });
    }
    condition.notify_one();
    return res;
}

inline bool ThreadPool::isEmpty() { return workers.empty(); }

inline void ThreadPool::clearPool()
{
    if (!isEmpty())
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
    workers.clear();
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    if (!isEmpty())
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }
}
} // namespace CPJ

class ExampleClass
{
  public:
    ExampleClass()
    {
        std::cout << "In ExampleClass, id = " << std::this_thread::get_id() << std::endl;
        std::vector<int> coreId_vec(1, 12);
        CPJ::ThreadPool tp(1, coreId_vec);
        tp.enqueue(&ExampleClass::example_method, this, 1, 0.1, "hello");
        Msg_check("Before clear: %d", tp.isEmpty());
        tp.clearPool();
        Msg_check("After clear: %d", tp.isEmpty());
    }
    void example_method(int x, float y, const std::string& z)
    {
        std::cout << "In example_method, id = " << std::this_thread::get_id() << std::endl;
        std::cout << "x: " << x << ", y: " << y << ", z: " << z << ", c = " << c << std::endl;
    }

  private:
    int c = 3;
};

#endif