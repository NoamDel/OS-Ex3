
#include <atomic>
#include <utility>
#include <pthread.h>
#include "MapReduceClient.h"
#include "MapReduceFramework.h"
#include "MapReduceClient.h"
#include <iostream>
#include <unistd.h>


struct ThreadContext; // forward
using namespace std;

#define SYSTEM_ERROR_PREFIX "system error: "
#define NEW_THREAD_ERR "cannot create new thread!"
#define WAKE_THREADS_ERR "cannot wake threads"
#define UNLOCK_MUTEX_ERR "cannot unlock mutex"
#define LOCK_MUTEX_ERR "cannot lock mutex"
#define CONDITIONAL_WAIT_ERR "cannot conditionally wait"
#define JOIN_THREAD_ERR "error joining thread"

// holds all ''globals''  for current job
/**
 * The main class in the program which returns by StartMapReduce function. It includes all the
 * variables needed to control the flow of the program without race condition.
 */
class jobInfo
{

public:
    jobInfo(const InputVec &inputVec, OutputVec &outputVec, int numThreads,
            const MapReduceClient &client) try :
            inputVec(inputVec),
            outputVec(outputVec),
            numThreads(numThreads),
            client(client),
            threads(new pthread_t[numThreads]),
            input_size(inputVec.size()),
            mutex(PTHREAD_MUTEX_INITIALIZER),
            cv(PTHREAD_COND_INITIALIZER),
            atomic_state(new std::atomic<uint64_t>(0)),
            maps_started_counter(new std::atomic<uint32_t>(0)),
            reduce_started_counter(new std::atomic<uint32_t>(0)),
            keys(new std::vector<K2 *>),
            thCs(new std::vector<ThreadContext *>),
            emit2Outputs(new IntermediateMap),
            barrier(new pthread_barrier_t)
    {
        pthread_barrier_init(barrier, nullptr, numThreads);
        for (int i = 0; i < numThreads - 1; ++i)
        {
            auto *m = new pthread_mutex_t;
            pthread_mutex_init(m, nullptr);
            mapMutexes.push_back(m);
        }
    }
    catch (bad_alloc &e)
    {
        cerr << e.what() << endl;
        exit(EXIT_FAILURE);
    }

    ~jobInfo()
    {
        if (finished)
        {
            delete[] threads;
            threads = nullptr;
            delete (atomic_state);
            atomic_state = nullptr;
            delete (maps_started_counter);
            maps_started_counter = nullptr;
            delete (reduce_started_counter);
            reduce_started_counter = nullptr;
            delete (keys);
            keys = nullptr;
            delete (emit2Outputs);
            emit2Outputs = nullptr;
            delete (thCs);
            thCs = nullptr;
            for (int i = 0; i < numThreads - 1; ++i)
            {
                delete (mapMutexes.at(i));
                mapMutexes.at(i) = nullptr;
            }
        }
    }

//private:
    const InputVec &inputVec;
    OutputVec &outputVec;
    int numThreads;
    const MapReduceClient &client;
    pthread_t *threads;
    long input_size;
    pthread_mutex_t mutex;
    pthread_cond_t cv;
    std::atomic<uint64_t> *atomic_state{};
    std::atomic<uint32_t> *maps_started_counter{};
    std::atomic<uint32_t> *reduce_started_counter;
    std::vector<K2 *> *keys;

    std::vector<ThreadContext *> *thCs;
    IntermediateMap *emit2Outputs;
    std::vector<pthread_mutex_t *> mapMutexes;
    pthread_barrier_t *barrier;
    bool finished = false;

};


// global info + info special to specific thread
/**
 * Context that each thread receives when created.
 */
struct ThreadContext
{
    jobInfo *_info;
    int id;
    std::vector<IntermediatePair> intermediateVec;

    ThreadContext(jobInfo *info, int id) : _info(info), id(id)
    {}
};

/**
 * This function preforms the "REDUCE" phase in the program. It calls the reduce function of the
 * client.
 * @param tc - Thread context instance which kept as a context in every thread.
 */
void letsReduce(ThreadContext *tc)
{

    jobInfo ji = *(tc->_info);
    unsigned long old_value;
    long size = ji.keys->size();
    //Reduce loop
    while (true)
    {
        old_value = (unsigned long) (*(ji.reduce_started_counter))++;
        if (old_value >= size)
        {
            break;
        }
        K2 *k2 = ji.keys->at(old_value);
        std::vector<V2 *> &v = (*ji.emit2Outputs)[k2];
        (ji.client).reduce(k2, v, tc);

        old_value = (*(ji.atomic_state))++; // first 31 bits are  for finished in current state.
        (void) old_value;  // to ignore not used warning
    }

}

/**
 * This method preforms the mapping which each thread using during "MAP" phase. The method calls
 * map function of the client.
 * @param arg - Thread context instance which kept as a context in every thread.
 * @return - The function doesn't return.
 */
// give this to each thread
void *threadRoutine(void *arg)
{
    auto *tc = (ThreadContext *) arg;
    jobInfo ji = *(tc->_info);
    unsigned long old_value;

    //map loop
    while (true)
    {
        // so each thread will get different maps:
        old_value = (unsigned long) (*(ji.maps_started_counter))++;
        if (old_value >= ji.input_size)
        {
            break;
        }
        const InputPair &ip = (ji.inputVec)[old_value];
        (ji.client).map(ip.first, ip.second, tc);
        // update finished map:
        (*(ji.atomic_state))++;
    }

    pthread_barrier_wait(ji.barrier); // waiting for shuffle.

    //Reduce
    letsReduce(tc);
    return arg;

}

/**
 * This function preforms the "SHUFFLE" part of the program. A single thread enter this method
 * and goes over each of the other threads emit2 output vectors and mapping each element according
 * to it's K2 value into an intermidiate map.
 * @param arg - Thread context instance which kept as a context in every thread.
 */
void *shuffleRoutine(void *arg)
{
    auto *tc = (ThreadContext *) arg;
    jobInfo ji = *(tc->_info);
    JobState state;
    std::vector<IntermediatePair> vec;
    auto *thCs = ji.thCs;
    auto mapMutexes = ji.mapMutexes;
    int shuffledCounter = 0;

    while (true)
    {
        //check for non-empty mappings, anf shuffle them. safely.
        for (int i = 0; i < ji.numThreads - 1; ++i)
        {
            if (pthread_mutex_lock((mapMutexes).at(i)) != 0)
            {
                cerr << SYSTEM_ERROR_PREFIX << LOCK_MUTEX_ERR << endl;
                exit(EXIT_FAILURE);
            }
            vec = (thCs->at(i))->intermediateVec;
            if (!vec.empty())
            {
                for (auto &j : vec)
                {
                    (*ji.emit2Outputs)[j.first].push_back(j.second);
                    shuffledCounter++; // can't update state yet, it is still mapping.
                }
                //  now erase shuffled pairs:
                while (!(thCs->at(i))->intermediateVec.empty())
                {
                    (thCs->at(i))->intermediateVec.pop_back();
                }
            }
            if (pthread_mutex_unlock((mapMutexes).at(i)) != 0)
            {
                cerr << SYSTEM_ERROR_PREFIX << UNLOCK_MUTEX_ERR << endl;
                exit(EXIT_FAILURE);
            }
        }
        getJobState(&ji, &state);
        if (state.percentage >= 100)
        { break; } // starting SHUFFLE officially.
    }


//    update state to SHUFFLE:
    uint64_t s = 0;
//    count total keys to map:
    uint64_t total = shuffledCounter;
    for (int i = 0; i < ji.numThreads - 1; ++i)
    {
        total += (thCs->at(i))->intermediateVec.size();
    }
    s += ((uint64_t(2)) << 62); // shuffle stage
    s += ((uint64_t(total)) << 31);
    (*(ji.atomic_state)) = s;
    getJobState(&ji, &state);


    // do shuffle:
    for (int i = 0; i < ji.numThreads - 1; ++i)
    {
        vec = (thCs->at(i))->intermediateVec;
        for (auto &j : vec)
        {
            (*ji.emit2Outputs)[j.first].push_back(j.second);
            //update state:
            (*(ji.atomic_state))++; // first 31 bits are  for finished in current state.
        }
    }

    for (auto const &x : (*ji.emit2Outputs))
    {
        ji.keys->push_back(x.first);
    }
    unsigned long reduceSize = ji.keys->size();

    //    update state to REDUCE:
    s = 3;
    (*(ji.atomic_state)) = (s << 62) + (reduceSize << 31);

    //awaken:
    pthread_barrier_wait(ji.barrier);
    //join the reducers..
    letsReduce(tc);
    return arg;
}


JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel)
{

    auto *ji = new jobInfo(inputVec, outputVec, multiThreadLevel, client);
    uint64_t s = 0;
    s += ((uint64_t(1)) << 62);
    s += ((uint64_t(inputVec.size())) << 31);
    (*(ji->atomic_state)) = s;

    for (int i = 0; i < multiThreadLevel - 1; ++i) // w\o shuffle thread.
    {
        auto thC = new ThreadContext(ji, i);
        ji->thCs->push_back(thC);
        if (pthread_create(ji->threads + i, nullptr, threadRoutine, thC) != 0)
        {
            cerr << SYSTEM_ERROR_PREFIX << NEW_THREAD_ERR << endl;
            exit(EXIT_FAILURE);
        }
    }
    // shuffle thread
    auto thC = new ThreadContext(ji, multiThreadLevel - 1);
    ji->thCs->push_back(thC);
    if (pthread_create(ji->threads + (multiThreadLevel - 1), nullptr, shuffleRoutine,
                       ji->thCs->at(multiThreadLevel - 1)) != 0)
    {
        cerr << SYSTEM_ERROR_PREFIX << NEW_THREAD_ERR << endl;
        exit(EXIT_FAILURE);
    }
    return (JobHandle) ji;


}


void getJobState(JobHandle job, JobState *state)
{
    const jobInfo *ji = (jobInfo *) job;
    stage_t stage;
    uint64_t atomicState = (*(ji->atomic_state)).load(); // only accessing so don't have to worry about atomic.
    switch (atomicState >> 62) // last 2 bits of atomic is state
    {
        case 0:
            stage = UNDEFINED_STAGE;
            break;
        case 1:
            stage = MAP_STAGE;
            break;
        case 2:
            stage = SHUFFLE_STAGE;
            break;
        case 3:
            stage = REDUCE_STAGE;
            break;
        default:
            stage = UNDEFINED_STAGE;
            break;
    }
    // Bits 1-31 are for number number of threads that have been processed and
    // 32-62 for overall threads
    auto percentage = (float) (atomicState & (0x7fffffff));
    auto total = (atomicState >> 31) & (0x7fffffff);
    percentage = percentage * 100 / total;
    (*state).stage = stage;
    (*state).percentage = percentage;
}

void emit2(K2 *key, V2 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    const IntermediatePair &pair = {key, value};
    if (pthread_mutex_lock(((tc->_info->mapMutexes)).at(tc->id)) != 0)
    {
        cerr << SYSTEM_ERROR_PREFIX << LOCK_MUTEX_ERR << endl;
        exit(EXIT_FAILURE);
    }// gotta be safe..
    tc->intermediateVec.push_back(pair);
    if (pthread_mutex_unlock(((tc->_info->mapMutexes)).at(tc->id)) != 0)
    {
        cerr << SYSTEM_ERROR_PREFIX << UNLOCK_MUTEX_ERR << endl;
        exit(EXIT_FAILURE);
    }
}

void emit3(K3 *key, V3 *value, void *context)
{
    auto *tc = (ThreadContext *) context;
    const OutputPair &pair = {key, value};
    if (pthread_mutex_lock(&(tc->_info->mutex)) != 0)
    {
        cerr << SYSTEM_ERROR_PREFIX << LOCK_MUTEX_ERR << endl;
        exit(EXIT_FAILURE);
    }
    tc->_info->outputVec.push_back(pair);
    if (pthread_mutex_unlock(&(tc->_info->mutex)) != 0)
    {
        cerr << SYSTEM_ERROR_PREFIX << UNLOCK_MUTEX_ERR << endl;
        exit(EXIT_FAILURE);
    }
}

void waitForJob(JobHandle job)
{
    auto *tc = (jobInfo *) job;
    for (int i = 0; i < tc->numThreads; i++)
    {
        if (pthread_join(tc->threads[i], nullptr) != 0)
        {
            cerr << SYSTEM_ERROR_PREFIX << JOIN_THREAD_ERR << endl;
            exit(EXIT_FAILURE);
        }
    }
}

void closeJobHandle(JobHandle job)
{
    auto *ji = (jobInfo *) job;
    waitForJob(ji);
    ji->finished = true;
    ji->~jobInfo();
}