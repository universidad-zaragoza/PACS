---
papersize: a4
lang: en
title: Problem Collection
subtitle: Programming and Architecture of Computing Systems
date: \today
geometry: "left=2cm,right=1.5cm,top=2cm,bottom=1.5cm"
urlcolor: blue
fontsize: 10pt
mainfont: Fira Sans
monofont: Fira Code
monofontoptions: 'Scale=0.9'
link-citations: true
hyperrefoptions:
- linktoc=all
header-includes: |
    \usepackage{fancyhdr}
    \pagestyle{fancy}
    \fancyhead[LO,LE]{PACS}
    \fancyhead[RO,RE]{Universidad de Zaragoza}
    \fancyfoot[CE,CO]{\thepage}
    \usepackage{fvextra}
    \DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
---

# Preliminary Notes

This brief collection of problems is divided in three parts. The first part covers
small questions, the second part includes test questions, and the third part refers to some longer exercises.

To report erratas, typos... please mail either [alvabre@unizar.es](mailto:alvabre@unizar.es),
[rgran@unizar.es](mailto:rgran@unizar.es) or [dario@unizar.es](mailto:dario@unizar.es).

# Small Questions

1. Please briefly respond to the following questions: ¿Is a concurrent
   application always parallel? ¿Is a parallel application always concurrent?

2. According to Amdahl's Law, for a program where the sequential part
   represents the 15% of the total, what would be the potential speed-up for a
   16-core machine.

3. Can a processor execute instructions from two different instruction sets?

4. Enumerate what are the key design features of GPUs to allow a very fast context switching of wavefronts.

5. Explain what the conditional branching problem is on GPUs and how it is solved.

6. Make comparative analysis between a GPU and an ASIC.

# Test Questions

1. The Local Data Share (LDS) cache on a GPU is used to:

	a) Amplify the regular cache bandwidth

	b) Execute atomic instructions

	c) Synchronization of wavefronts

	d) All of the above

	e) None of the above

2. In OpenCL, Local Memory is shared between:

	a) All the workitems of a global work domain

	b) Workitems in the same kernel launch

	c) Local Memory is an abstraction not present in OpenCL

	d) Workitems in the same workgroup

	e) None of the previous ones

3. In OpenCL, workitems that access global shared variable must explicitly assure memory order in order to avoid race conditions

	a) Always

	b) Just in case they do not belong to the same workgroup

	c) Just in case they do belong to the same workgroup

	d) Never

# Exercises

1. The dot product algorithm takes two vectors of the same length and returns a
   single number. The number is the sum of the products of the corresponding
   entries in the input victors.

    In C++, the algorithm can be coded as follows:

    ```C++
    template<typename T>
    T doc_product(const std::vector<T> &a, const std::vector<T> &b>
    {

    if(a.lenght() != b.lenght()) {
        error(...);
    }

    // initialize to 0 regardless the type
    T dot_p{}; // Also T dot_p = T();

    for(size_t i = 0; i < a.length(); ++i) {
      dot_p+=(a[i]*b[i]);
    }
    ```

    Please answer the following questions:

    a. Implement the doc product using threads and static partitioning.

    b. Implement the doc product assuming you have the thread pool and the
       thread-safe queue from Laboratory 4.

    c. For the thread-pool version, would all tasks perform the same ammount of work?

2. Given an `std::vector<int>` array, could you please write a parallel
   algorithm that finds the minimum and maximum values of the array.

3. See Exercise 2 from the collection of exercises referring to metrics.

4. Write an OpenCL program that calculates the dot product of to integer
   arrays. Additionally to the kernel code, in the host side of the program,
   just focus on the buffer management, command-queue management and kernel
   launch.

    a. Please, analytically model the execution time of this work assuming the
    computational device has the following characteristics: 8 compute units,
    each compute unit has 128 parallel cores, each core has two floating-point
    arithmetic units and, frequency of the computational device is 1.5GHz.
    Assumption 1: just floating point instructions contribute to the execution
    time. Assumption 2: each FPU can process a floating point instrution per
    cycle.

5. Please write a parallel program that given an array of integer values,
   it finds those values that are prime and larger than a given element. The solution
   should follow a fork-join parallelism model in C++. To know whether an integer
   value is prime, you can assume that the function `bool is_prime(int n)` is
   available:

    ```C++
    bool is_prime(int n) {
      if (n <= 3) {
        return n > 1;
      }

      if (((n % 2) == 0) || (( n % 3) == 0)) {
        return false;
      }

      for(int i = 5; i*i <= n; i+=6) {
        return false;
      }
      return true;
    }
    ```

6. Sorting is one of the most important problem in computing.
   Its computational intensity makes sorting an ideal candidate for
   parallelization.  One of the most common implementation is bucket sort where the
   input array is split between N buckets that are independently sorted and then
   concatenated.

   ![Bucket sort example](./figures/bucket_sort.pdf ){ height=150 }

     The upper figure shows bucket sort main steps: 1) Computation of the ranges.
     With the maximum value stored in the input array and the number of buckets,
     you compute the range, 63/3=21 in the example, and create the buckets. 2)
     Bucket insertion. Each value of the array goes to its bucket. 3) Sorting
     buckets. Each bucket is sortened independently, and 4) the sorted array is
     assembled by concatenating the arrays of each bucket.

     a. Please implement a parallel version of the buckle sort algorithm. For sorting
     the buckets, you can use any standard sequential sorting algorithm as

         ```C++
         template<typename T>
         void insertion_sort(std::vector<T>& array)
         {
           for(size_t i = 1; i < array.size(); ++i) {
             for(size_t j = i; (j > 0) && (array[j-1] > array[j]); --j) {
               std::swap(array[j], array[j-1]);
             }
           }
         }
         ```

     b. ¿Is there any phatological case where the parallel version could not be
        faster than the sequential version?

   Notes: You can concatenate two `std::vector` arrays with the `insert`
   method; e.g., `dst.insert( dst.end(), src.begin(), src.end())`. The function
   `float std::floor(float arg)` computes the largest integer value not greater
   than `arg`.

7. Many Computer Vision applications require the computation of
histograms, which help to understand the distributions of a set of numbers.
To compute an histogram, you need to visit all elements of the set, compute
their bucket, and then increase the corresponding counter of that bucket.
For example, if the input array contains this set of numbers {0, 1, 1, 1, 2,
3}, the output histogram with 4 buckets will be {1, 3, 1, 1}.

   a. Please write a parallel version of an histogram for integer values following
   a fork-join approximation in C++. The histogram result will be stored in an array
   of atomic variables and you have to minimize contention on this array.

   You can assume the following initial skeleton:

   ```C++
   int main() {

       const size_t N = 1024*8; // array size
       const size_t m_buckets = 32; // buckets
       const size_t n_threads = 8;

       std::vector<int> array; // please asume this array has been already initialized
       std::vector<std::atomic<init>> histogram(m_buckets);

       // ...
    ```

    b. What could be the maximum speed-up of this implementation?
