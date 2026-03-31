# OPERATING SYSTEM

## CT 656

### Lecture : 3 Year : III
### Tutorial : 1 Part : II
### Practical : 1

```
Course Objective:
The objective of the course is to be familiar with the different aspects of
operating system and use the idea in designing operating system.
```

**1. Introduction [5 hours]**
1.1. Operating System and Function  
1.2. Evolution of Operating System  
1.3. Type of Operating System: Batch, Interactive, Multiprocessing, Time Sharing and Real Time System  
1.4. Operating System Components  
1.5. Operating System Structure: Monolithic, Layered, Micro-Kernel, Client-Server, Virtual Machine  
1.6. Operating System Services  
  1.6.1. System calls  
  1.6.2. Shell commands  
  1.6.3. Shell programming  
1.7. Examples of O.S.: UNIX, Linux, MS-Windows, Handheld OS  

**2. Process Management [6 hours]**
2.1. Introduction to Process  
  2.1.1. Process description  
  2.1.2. Process states  
  2.1.3. Process control  
2.2. Threads  
2.3. Processes and Threads  
2.4. Scheduling  
  2.4.1. Types of scheduling  
  2.4.2. Scheduling in batch system  
  2.4.3. Scheduling in interactive system  
  2.4.4. Scheduling in real-time system  
  2.4.5. Thread scheduling  
2.5. Multiprocessor scheduling concept  

**3. Process Communication and Synchronization [5 hours]**
3.1. Principles of Concurrency  
3.2. Critical Region  
3.3. Race Condition  
3.4. Mutual Exclusion  
3.5. Semaphores and Mutex  
3.6. Message Passing  
3.7. Monitors  
3.8. Classical Problems of Synchronization:  
  - Readers-Writers Problem  
  - Producer Consumer Problem  
  - Dining Philosopher Problem  

**4. Memory Management [6 hours]**
4.1. Memory address, Swapping and Managing Free Memory Space  
4.2. Resident Monitor  
4.3. Multiprogramming with Fixed Partition  
4.4. Multiprogramming with Variable Partition  
4.5. Multiple Base Register  
4.6. Virtual Memory Management  
  4.6.1. Paging  
  4.6.2. Segmentation  
  4.6.3. Paged Segmentation  
4.7. Demand Paging  
4.8. Performance  
4.9. Page Replacement Algorithms  
4.10. Allocation of Frames  
4.11. Thrashing  

**5. File Systems [6 hours]**
5.1. File: Name, Structure, Types, Access, Attribute, Operations  
5.2. Directory and File Paths  
5.3. File System Implementation  
  5.3.1. Selecting Block Size  
  5.3.2. Impact of Block Size Selection  
  5.3.3. Implementing File: Contiguous Allocation, Linked List Allocation, Indexed Allocation (Inode)  
  5.3.4. Implementing Directory  
5.4. Impact of Allocation Policy on Fragmentation  
5.5. Mapping File Blocks on the Disk  
5.6. File System Performance  
5.7. Example File Systems: CD-ROM, MS-DOS, UNIX  

**6. I/O Management & Disk Scheduling [4 hours]**
6.1. Principles of I/O Hardware  
6.2. Principles of I/O Software  
6.3. I/O Software Layer  
6.4. Disk  
  6.4.1. Hardware  
  6.4.2. Formatting  
  6.4.3. Arm scheduling  
  6.4.4. Error handling  
  6.4.5. Stable storage  

**7. Deadlock [5 hours]**
7.1. Principles of Deadlock  
7.2. Deadlock Prevention  
7.3. Deadlock Avoidance  
7.4. Deadlock Detection  
7.5. Recovery from Deadlock  
7.6. Integrated Deadlock Strategies  
7.7. Other Issues: Two-phase locking, Communication Deadlock, Livelock, Starvation  

**8. Security [4 hours]**
8.1. Security Breaches  
8.2. Types of Attacks  
8.3. Security Policy and Access Control  
8.4. Basics of Cryptography  
8.5. Protection Mechanisms  
8.6. Authentication  
8.7. OS Design Considerations for Security  
8.8. Access Control Lists and OS Support  

**9. System Administration [4 hours]**
9.1. Administration Tasks  
9.2. User Account Management  
9.3. Start and Shutdown Procedures  
9.4. Setting up Operational Environment for a New User  
9.5. AWK tool, Search, Sort tools, Shell Scripts, Make tool  


```
Practical:
```
1. Shell commands and shell programming (functions, loops, patterns, substitutions)  
2. Programs using UNIX system calls: fork, exec, getpid, exit, wait, close, stat, opendir, readdir  
3. Programs using I/O system calls  
4. Implement Producer–Consumer problem using semaphores  
5. Implement memory management schemes  


```
Reference Books:
```
1. Andrew S. Tanenbaum, *Modern Operating Systems*, PHI  
2. William Stallings, *Operating Systems*, Pearson  
3. Silberschatz, Galvin, Gagne, *Operating System Concepts*, Wiley  
4. Milan Milenkovic, *Operating Systems Concepts and Design*, TMGH  
5. Sumitabha Das, *Unix Concepts and Applications*, Tata McGraw Hill  
6. M. J. Bach, *The Design of the UNIX Operating System*, PHI  
7. Charles Crowley, *Operating Systems: A Design-Oriented Approach*, TMH  


```
Evaluation Scheme:
```
Unit 1: Introduction | 5 Hours | 10 Marks  
Unit 2: Process Management | 6 Hours | 10 Marks  
Unit 3: Process Communication and Synchronization | 5 Hours | 10 Marks  
Unit 4: Memory Management | 6 Hours | 10 Marks  
Unit 5: File Systems | 6 Hours | 10 Marks  
Unit 6: I/O Management & Disk Scheduling | 4 Hours | 12 Marks  
Unit 7: Deadlock | 5 Hours | (Included)  
Unit 8: Security | 4 Hours | 12 Marks  
Unit 9: System Administration | 4 Hours | (Included)  

Total: 45 Hours | 80 Marks  

There can be minor deviations in the numbers.
```