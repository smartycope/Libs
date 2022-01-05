#ifndef __COPE__
#define __COPE__


// This file doesn't currently work, I don't think

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <stdarg.h>

//* I'm not sure why this doesn't work...
// #ifdef __PRETTY_FUNCTION__
//     #define __FUNC__  __PRETTY_FUNCTION__
// #else
//     #ifdef __FUNCTION__
//         #define __FUNC__  __FUNCION__
//     #else
//         #define __FUNC__  __func__
//     #endif
// #endif


#define __FUNC__     __PRETTY_FUNCTION__
// #define __FUNC__  __FUNCION__
// #define __FUNC__  __func__


// #define METADATA_FORMAT_STANDARD
#define METADATA_FORMAT_VSCODE

#define PRINT_FILE
#define PRINT_FUNC
#define PRINT_LINE
// #define PRINT_FULL_LINK
// #define PRINT_SHORT_LINK

#define DEBUG_COUNT_COLOR   COLOR_RESET
#define METADATA_COLOR      COLOR_GREEN
#define DEBUG_COLOR         COLOR_BLUE
#define LINK_COLOR          COLOR_HIDDEN
#define REGULAR_DEBUG_COLOR COLOR_CYAN
#define TODO_COLOR          COLOR_YELLOW


#define MAX_STR_LEN 1024

#ifndef __THIS_THING__
#define __THIS_THING__
static unsigned long debugCount = 0;
#endif

//* Colors
#define COLOR_RED     (char*)"\x1b[31m"
#define COLOR_GREEN   (char*)"\x1b[32m"
#define COLOR_YELLOW  (char*)"\x1b[33m"
#define COLOR_BLUE    (char*)"\x1b[34m"
#define COLOR_MAGENTA (char*)"\x1b[35m"
#define COLOR_CYAN    (char*)"\x1b[36m"
#define COLOR_RESET   (char*)"\x1b[0m"
#define COLOR_HIDDEN  (char*)"\x1b[30m"


#define CC(a, b) __CONCAT(a, b)

#ifndef __STRING
    #define __STRING(a) #a
#endif
#define UNPACK(...) __VA_ARGS__
#define COMMA ,

//! Depricated
// static char* color(char*, char*);
// static char* _fmt(char*, ...);

static void  _printMetaData(int, char*, char*);
static void  printDebugCount();
static void printfclr(const char*, const char*, ...);
static void printclr(const char*, const char*);
static void printfclrsl(const char*, const char*, ...);
static void printclrsl(const char*, const char*);
static void leave(int status, const char* msg);
static void leave(int status);
// std::vector<char> getCmdOutV(const std::string command, bool verbose = false, bool exitOnFail = true);
// static std::string getCmdOutS(const std::string command, bool verbose = false, bool exitOnFail = true);

// static std::string color(std::string, std::string);
// static std::string fmt(std::string, ...);



// I LIKE these.
#ifndef true
#define true  1
#endif
#ifndef false
#define false 0
#endif
#ifndef byte
#define byte  unsigned char
#endif

#ifndef not
#define not !
#endif
#ifndef or
#define or  ||
#endif
#ifndef and
#define and &&
#endif

#define fmt(str, ...) _fmt((char*)str, __VA_ARGS__)

#define __LIST_OF_TYPES__ \
    void *:                   "void*",                   \
    _Bool:                    "bool",                    \
    _Bool *:                  "bool*",                   \
    char:                     "char",                    \
    char *:                   "char*",                   \
    signed char:              "signed char",             \
    signed char *:            "signed char*",            \
    unsigned char:            "unsigned char",           \
    unsigned char *:          "unsigned char*",          \
    short int:                "short",                   \
    short int *:              "short*",                  \
    unsigned short int:       "unsigned short",          \
    unsigned short int *:     "unsigned short*",         \
    int:                      "int",                     \
    int *:                    "int*",                    \
    unsigned int:             "unsigned int",            \
    unsigned int *:           "unsigned int*",           \
    long int:                 "long int",                \
    long int *:               "long int*",               \
    unsigned long int:        "unsigned long int",       \
    unsigned long int *:      "unsigned long int*",      \
    long long int:            "long long int",           \
    long long int *:          "long long int*",          \
    unsigned long long int:   "unsigned long long int",  \
    unsigned long long int *: "unsigned long long int*", \
    float:                    "float",                   \
    float*:                   "float*",                  \
    double:                   "double",                  \
    double*:                  "double*",                 \
    long double:              "long double",             \
    long double*:             "long double*",            \
                                                        \
    const void *:                   "const void*",                   \
    const _Bool:                    "const bool",                    \
    const _Bool *:                  "const bool*",                   \
    const char:                     "const char",                    \
    const char *:                   "const char*",                   \
    const signed char:              "const signed char",             \
    const signed char *:            "const signed char*",            \
    const unsigned char:            "const unsigned char",           \
    const unsigned char *:          "const unsigned char*",          \
    const short int:                "const short",                   \
    const short int *:              "const short*",                  \
    const unsigned short int:       "const unsigned short",          \
    const unsigned short int *:     "const unsigned short*",         \
    const int:                      "const int",                     \
    const int *:                    "const int*",                    \
    const unsigned int:             "const unsigned int",            \
    const unsigned int *:           "const unsigned int*",           \
    const long int:                 "const long int",                \
    const long int *:               "const long int*",               \
    const unsigned long int:        "const unsigned long int",       \
    const unsigned long int *:      "const unsigned long int*",      \
    const long long int:            "const long long int",           \
    const long long int *:          "const long long int*",          \
    const unsigned long long int:   "const unsigned long long int",  \
    const unsigned long long int *: "const unsigned long long int*", \
    const float:                    "const float",                   \
    const float*:                   "const float*",                  \
    const double:                   "const double",                  \
    const double*:                  "const double*",                 \
    const long double:              "const long double",             \
    const long double*:             "const long double*",            \
                                                                                    \
    const volatile void *:                   "const volatile void*",                   \
    const volatile _Bool:                    "const volatile bool",                    \
    const volatile _Bool *:                  "const volatile bool*",                   \
    const volatile char:                     "const volatile char",                    \
    const volatile char *:                   "const volatile char*",                   \
    const volatile signed char:              "const volatile signed char",             \
    const volatile signed char *:            "const volatile signed char*",            \
    const volatile unsigned char:            "const volatile unsigned char",           \
    const volatile unsigned char *:          "const volatile unsigned char*",          \
    const volatile short int:                "const volatile short",                   \
    const volatile short int *:              "const volatile short*",                  \
    const volatile unsigned short int:       "const volatile unsigned short",          \
    const volatile unsigned short int *:     "const volatile unsigned short*",         \
    const volatile int:                      "const volatile int",                     \
    const volatile int *:                    "const volatile int*",                    \
    const volatile unsigned int:             "const volatile unsigned int",            \
    const volatile unsigned int *:           "const volatile unsigned int*",           \
    const volatile long int:                 "const volatile long int",                \
    const volatile long int *:               "const volatile long int*",               \
    const volatile unsigned long int:        "const volatile unsigned long int",       \
    const volatile unsigned long int *:      "const volatile unsigned long int*",      \
    const volatile long long int:            "const volatile long long int",           \
    const volatile long long int *:          "const volatile long long int*",          \
    const volatile unsigned long long int:   "const volatile unsigned long long int",  \
    const volatile unsigned long long int *: "const volatile unsigned long long int*", \
    const volatile float:                    "const volatile float",                   \
    const volatile float*:                   "const volatile float*",                  \
    const volatile double:                   "const volatile double",                  \
    const volatile double*:                  "const volatile double*",                 \
    const volatile long double:              "const volatile long double",             \
    const volatile long double*:             "const volatile long double*",            \
                                                                        \
    volatile void *:                   "volatile void*",                   \
    volatile _Bool:                    "volatile bool",                    \
    volatile _Bool *:                  "volatile bool*",                   \
    volatile char:                     "volatile char",                    \
    volatile char *:                   "volatile char*",                   \
    volatile signed char:              "volatile signed char",             \
    volatile signed char *:            "volatile signed char*",            \
    volatile unsigned char:            "volatile unsigned char",           \
    volatile unsigned char *:          "volatile unsigned char*",          \
    volatile short int:                "volatile short",                   \
    volatile short int *:              "volatile short*",                  \
    volatile unsigned short int:       "volatile unsigned short",          \
    volatile unsigned short int *:     "volatile unsigned short*",         \
    volatile int:                      "volatile int",                     \
    volatile int *:                    "volatile int*",                    \
    volatile unsigned int:             "volatile unsigned int",            \
    volatile unsigned int *:           "volatile unsigned int*",           \
    volatile long int:                 "volatile long int",                \
    volatile long int *:               "volatile long int*",               \
    volatile unsigned long int:        "volatile unsigned long int",       \
    volatile unsigned long int *:      "volatile unsigned long int*",      \
    volatile long long int:            "volatile long long int",           \
    volatile long long int *:          "volatile long long int*",          \
    volatile unsigned long long int:   "volatile unsigned long long int",  \
    volatile unsigned long long int *: "volatile unsigned long long int*", \
    volatile float:                    "volatile float",                   \
    volatile float*:                   "volatile float*",                  \
    volatile double:                   "volatile double",                  \
    volatile double*:                  "volatile double*",                 \
    volatile long double:              "volatile long double",             \
    volatile long double*:             "volatile long double*",            \
    \
    default: "unknown"
//

#define _typename(x) _Generic((x), __LIST_OF_TYPES__)

#define printf_dec_format(x) _Generic((x), \
    char: "%c", \
    signed char: "%hhd", \
    unsigned char: "%hhu", \
    signed short: "%hd", \
    unsigned short: "%hu", \
    signed int: "%d", \
    unsigned int: "%u", \
    long int: "%ld", \
    unsigned long int: "%lu", \
    long long int: "%lld", \
    unsigned long long int: "%llu", \
    float: "%f", \
    double: "%f", \
    long double: "%Lf", \
    char *: "%s", \
    void *: "%p")
//

#define print(x)   printf(printf_dec_format(x), x), nl();
#define printsl(x) printf(printf_dec_format(x), x);

#define nl() puts((char*)"");


#ifdef PRINT_FULL_LINK
    #define _printVscodeLink(file, line, func) printsl(color((char*)"\n\t",     LINK_COLOR)); \
                                               printsl(color((char*)"File ",   LINK_COLOR)); \
                                               printsl(color((char*)__STRING(file),      LINK_COLOR)); \
                                               printsl(color((char*)", line ", LINK_COLOR)); \
                                               printsl(color((char*)__STRING(line), LINK_COLOR)); \
                                               printsl(color((char*)", in ",   LINK_COLOR)); \
                                               printsl(color((char*)func,      LINK_COLOR));

    #define printVscodeLink() _printVscodeLink(__FILE__, __LINE__, __FUNC__)
#else
    #ifdef PRINT_SHORT_LINK
        #define _printVscodeLink(file, line, func) printsl(color("\n\t",     LINK_COLOR)); \
                                                   printsl(color(__STRING(file), LINK_COLOR)); \
                                                   printsl(color(", line ", LINK_COLOR)); \
                                                   printsl(color(__STRING(line), LINK_COLOR));

        #define printVscodeLink() _printVscodeLink(__FILE__, __LINE__, __FUNC__)
    #else
        #define printVscodeLink()
    #endif
#endif

// printsl(color(fmt(__STRING(File __STRING(__FILE__)[,] line __LINE__[,] in __func__)), LINK_COLOR))

// #define leave(exitStatus) { debugs(__STRING(Exiting with code exitStatus)); exit(exitStatus); }

// #define assert(test) printDebugCount(); printMetaData(); assert(test);

#define printMetaData() _printMetaData(__LINE__, (char*)__FUNC__, (char*)__FILE__)

#define debug_0()           shout
#define debug_1(var)        { printDebugCount(); printMetaData(); printfclrsl(DEBUG_COLOR, "%s %s = ", _typename(var), #var); printsl(var); nl(); }
#define debug_2(var, clr)   { printDebugCount(); printMetaData(); printfclrsl(fmt("\x1b[3%dm", clr).c_str(), "%s %s = ", _typename(var), #var); printsl(var); nl(); }
#define debugs(string)      { printDebugCount(); printMetaData(); printclr(REGULAR_DEBUG_COLOR, string); printVscodeLink(); }
#define debugf(string, ...) { printDebugCount(); printMetaData(); printfclr(REGULAR_DEBUG_COLOR, string, __VA_ARGS__); printVscodeLink(); }
#define shout               { printDebugCount(); printMetaData(); printclrsl(REGULAR_DEBUG_COLOR, (char*)"HERE! HERE!", 1); printVscodeLink(); }
#define note(id)            debugs(#id);
#define noteCall()          debugf("%s called!", __FUNC__);
#define notecall()          noteCall();
#define _debugs             debugs
#define _debugf             debugf
// #define debugged(type, var) __debugged<type>(var, (char*)#var, _typename(var))

#define debug_X(x, var, color, FUNC, ...) FUNC
#define _debug(...)  debug_X(,##__VA_ARGS__,\
                      debug_2(__VA_ARGS__),\
                      debug_1(__VA_ARGS__),\
                      debug_0(__VA_ARGS__) \
                    )
//

#define todo(feature) { printDebugCount(); printMetaData(); printclrsl(TODO_COLOR, "Todo: "); printclr(REGULAR_DEBUG_COLOR, feature); printVscodeLink(); }


static void _printMetaData(int line, char* func, char* file){
    printsl(METADATA_COLOR);
    // printsl("-")

    #ifdef METADATA_FORMAT_STANDARD
        #if defined PRINT_FILE || defined PRINT_FUNC || defined PRINT_LINE
            printsl("[");
        #endif
        #ifdef PRINT_FILE
            printsl(file);
            printsl("->");
        #endif
        #ifdef PRINT_FUNC
            printsl(func);
            printsl("->");
        #endif
        #ifdef PRINT_LINE
            printsl(line);
        #endif
        #if defined PRINT_FILE || defined PRINT_FUNC || defined PRINT_LINE
            printsl("] ");
        #endif
    #else
        #ifdef METADATA_FORMAT_VSCODE
            printsl("[\"");
            printsl(file);
            printsl("\", line ");
            printsl(line);
            #ifdef PRINT_FUNC
                printsl(", in ")
                printsl(func);
            #endif
            printsl("] ")
        #endif
    #endif

    printsl(COLOR_RESET);
}

static void printDebugCount(){
    ++debugCount;
    printfclrsl(DEBUG_COUNT_COLOR, "%lu: ", debugCount);
}

/*
static char* color(char* str, char* color){
    char* tmp = (char*)malloc(MAX_STR_LEN);
    strcat(tmp, color);
    strcat(tmp, str);
    strcat(tmp, COLOR_RESET);
    strcat(tmp, "\0");

    return tmp;
    // return strcat(strcat(color, str), COLOR_RESET);
}
*/

static char* _fmt(char* str, ...){
    // char* tmp;
    char* tmp = (char*)malloc(MAX_STR_LEN);
    // char tmp[MAX_PARAM_LEN];
    va_list args;
    va_start(args, str);
    // vsnprintf(tmp, MAX_PARAM_LEN, str, args);
    vsnprintf(tmp, MAX_STR_LEN, str, args);
    va_end(args);
    return tmp;
}

static void printfclr(const char* color, const char* str, ...){
    printsl(color);

    va_list args;
    va_start(args, str);
    vprintf(str, args);
    va_end(args);

    print(COLOR_RESET);
}

static void printfclrsl(const char* color, const char* str, ...){
    printsl(color);

    va_list args;
    va_start(args, str);
    vprintf(str, args);
    va_end(args);

    printsl(COLOR_RESET);
}

static void printclrsl(const char* color, const char* str){
    printsl(color);
    printsl(str);
    printsl(COLOR_RESET);
}

static void printclr(const char* color, const char* str){
    printsl(color);
    printsl(str);
    print(COLOR_RESET);
}

static std::string color(std::string str, std::string color){
    return color + str + COLOR_RESET;
}

static std::string fmt(std::string str, ...){
    // char* tmp;
    char* tmp = (char*)malloc(str.size() + 1);
    // char tmp[MAX_PARAM_LEN];
    va_list args;
    va_start(args, str);
    // vsnprintf(tmp, MAX_PARAM_LEN, str, args);
    vsnprintf(tmp, str.size(), str.c_str(), args);
    va_end(args);
    return tmp;
}

static void leave(int status, std::string msg){
    debugf("Exiting with code %d: %s", status, msg.c_str());
    exit(status);
}

static void leave(int status){
    leave(status, "");
}

static void debugArgv(int argc, char* argv[]){
    _debug(argc);
    for(int i=0;i<argc;i++){
        //* Just kidding, we WANT it to break
        // do{ // Poor man's try statement
            // if (not argv[i])
                // continue;
        print(std::string(argv[i]));
        // }while(0);
    }
}


static std::string cStr(std::string str){
    return str + '\0';
}

static std::string cStr(char* str){
    return std::string(str) + '\0';
}


inline std::vector<char> getCmdOutV(const std::string command, bool verbose, bool exitOnFail){
    FILE* pipe;
    const char* cmd = command.c_str();
    std::vector<char> out;
    char* transportBuffer;

    if (verbose)
        printf("Executing command: \"%s\"\n", cmd);

    pipe = popen(cmd, "r");
    if (pipe == NULL) { _debug("Error: Failed to open pipe", 5); }

    // long read = fread(transportBuffer, sizeof(char), ftell(pipe), pipe);
    int read = 0;
    for (int c = 1; c != EOF; read++){
        c = fgetc(pipe);
        out.push_back(c);
    }

    int rtn = pclose(pipe);
    if (rtn and exitOnFail) leave(rtn, fmt("\"%s\" returned error: %d", cmd, rtn));

    // out.assign(transportBuffer, transportBuffer + read);

    return out;
}

static std::string getCmdOutS(const std::string command, bool verbose, bool exitOnFail){
    FILE* pipe;
    const char* cmd = command.c_str();
    std::string out;
    char* transportBuffer;

    if (verbose)
        printf("Executing command: \"%s\"\n", cmd);

    pipe = popen(cmd, "r");
    if (pipe == NULL) { _debug("Error: Failed to open pipe", 5); }

    int read = 0;
    for (int c = 1; c != EOF; read++){
        c = fgetc(pipe);
        out.push_back(c);
    }

    int rtn = pclose(pipe);
    if (rtn and exitOnFail) leave(rtn, fmt("\"%s\" returned error: %d", cmd, rtn));

    out.pop_back();

    // long read = fread(transportBuffer, sizeof(char), ftell(pipe), pipe);
    // return std::string(transportBuffer, read);
    return out;
}

#endif // __COPE__



















//* Old code:

/*
typedef struct{
    int line;
    char* file;
    char* func;
}MetaData;


#define getMetaData() metaDataConstructor(__LINE__, (char*)__FILE__, (char*)__FUNC__)


MetaData metaDataConstructor(int line, char* file, char* func){
    MetaData data;
    data.line = line;
    data.file = file;
    data.func = func;
    return data;
}


void printMetaData(MetaData metadata){
    #if defined PRINT_FILE || defined PRINT_FUNC || defined PRINT_LINE
    print(METADATA_COLOR);
    printsl("[");
    #endif
    #ifdef PRINT_FILE
    printsl(metadata.file);
    printsl("->");
    #endif
    #ifdef PRINT_FUNC
    printsl(metadata.func);
    printsl("->");
    #endif
    #ifdef PRINT_LINE
    printsl(metadata.line);
    #endif
    #if defined PRINT_FILE || defined PRINT_FUNC || defined PRINT_LINE
    printsl("] ");
    print(COLOR_RESET);
    #endif
}
*/





// char* fmt(const char *f, ...){
//     // int error;
//     va_list ap;
//     char* strp;

//     va_start(ap, f);
//     // error = vasprintf(strp, fmt, ap);
//     // va_list ap1;
//     size_t size;
//     // char *buffer;

//     // va_copy(ap1, ap);
//     char* buffer = (char*)malloc(1);
//     // size = vsnprintf(buffer, MAX_STR_LEN, f, ap) + 1;
//     va_end(ap);

//     if (!buffer)
//         return NULL;

//     strp = buffer;
//     vsnprintf(buffer, size, f, ap);

//     return strp;
// }



#if 0

    namespace COPE{
    static unsigned long debugCount = 0;
    }




template<typename T> static T __debugged(T var, char* name, char* type){
    printDebugCount();
    printMetaData();
    printf("%s %s = ", type, name);
    printv(var);
    return var;
}





//static void quit(int exitStatus){
//    debugs("Exiting with code %d" % exitStatus);
//    exit(exitStatus);
//}




//* How to do optional-arguement macros
/*
// The multiple macros that you would need anyway
#define XXX_0()                     <code for no arguments>
#define XXX_1(A)                    <code for one argument>
#define XXX_2(A,B)                  <code for two arguments>
#define XXX_3(A,B,C)                <code for three arguments>
#define XXX_4(A,B,C,D)              <code for four arguments>

// The interim macro that simply strips the excess and ends up with the required macro
#define XXX_X(x,A,B,C,D,FUNC, ...)  FUNC

// The macro that the programmer uses
#define XXX(...)                    XXX_X(,##__VA_ARGS__,\
                                          XXX_4(__VA_ARGS__),\
                                          XXX_3(__VA_ARGS__),\
                                          XXX_2(__VA_ARGS__),\
                                          XXX_1(__VA_ARGS__),\
                                          XXX_0(__VA_ARGS__)\
                                         )
*/

//#undef cout





#ifndef DEBUG_H
#define DEBUG_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif /* HAVE_CONFIG_H */

/*
** Usage:  TRACE((level, fmt, ...))
** "level" is the debugging level which must be operational for the output
** to appear. "fmt" is a printf format string. "..." is whatever extra
** arguments fmt requires (possibly nothing).
** The non-debug macro means that the code is validated but never called.
** -- See chapter 8 of 'The Practice of Programming', by Kernighan and Pike.
*/
#ifdef DEBUG
#define TRACE(x)    db_print x
#else
#define TRACE(x)    do { if (0) db_print x; } while (0)
#endif /* DEBUG */

#ifndef lint
#ifdef DEBUG
/* This string can't be made extern - multiple definition in general */
static const char jlss_id_debug_enabled[] = "@(#)*** DEBUG ***";
#endif /* DEBUG */
#ifdef MAIN_PROGRAM
const char jlss_id_debug_h[] = "@(#)$Id: debug.h,v 3.6 2008/02/11 06:46:37 jleffler Exp $";
#endif /* MAIN_PROGRAM */
#endif /* lint */

#include <stdio.h>

extern int      db_getdebug(void);
extern int      db_newindent(void);
extern int      db_oldindent(void);
extern int      db_setdebug(int level);
extern int      db_setindent(int i);
extern void     db_print(int level, const char *fmt,...);
extern void     db_setfilename(const char *fn);
extern void     db_setfileptr(FILE *fp);
extern FILE    *db_getfileptr(void);

#endif
#endif
