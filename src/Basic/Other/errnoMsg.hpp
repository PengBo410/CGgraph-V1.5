#pragma once

#include <string>

inline std::string getErrorMsg(int errno_)
{
    std::string errorMessage = "[SUCCESS (0): No error]";

    if (errno_ == 0) return errorMessage;

    // clang-format off
        switch (errno_) 
        {
        case EPERM: errorMessage = "[EPERM (1)]: Operation not permitted"; break;
        case ENOENT: errorMessage = "[ENOENT (2)]: No such file or directory"; break;
        case ESRCH: errorMessage = "[ESRCH (3)]: No such process"; break;
        case EINTR: errorMessage = "[EINTR (4)]: Interrupted system call"; break;
        case EIO: errorMessage = "[EIO (5)]: I/O error"; break;
        case ENXIO: errorMessage = "[ENXIO (6)]: No such device or address"; break;
        case E2BIG: errorMessage = "[E2BIG (7)]: Argument list too long"; break;
        case ENOEXEC: errorMessage = "[ENOEXEC (8)]: Exec format error"; break;
        case EBADF: errorMessage = "[EBADF (9)]: Bad file number"; break;
        case ECHILD: errorMessage = "[ECHILD (10)]: No child processes"; break;
        case EAGAIN: errorMessage = "[EAGAIN (11)]: Try again"; break;
        case ENOMEM: errorMessage = "[ENOMEM (12)]: Out of memory"; break;
        case EACCES: errorMessage = "[EACCES (13)]: Permission denied"; break;
        case EFAULT: errorMessage = "[EFAULT (14)]: Bad address"; break;
        case ENOTBLK: errorMessage = "[ENOTBLK (15)]: Block device required"; break;
        case EBUSY: errorMessage = "[EBUSY (16)]: Device or resource busy"; break;
        case EEXIST: errorMessage = "[EEXIST (17)]: File exists"; break;
        case EXDEV: errorMessage = "[EXDEV (18)]: Cross-device link"; break;
        case ENODEV: errorMessage = "[ENODEV (19)]: No such device"; break;
        case ENOTDIR: errorMessage = "[ENOTDIR (20)]: Not a directory"; break;
        case EISDIR: errorMessage = "[EISDIR (21)]: Is a directory"; break;
        case EINVAL: errorMessage = "[EINVAL (22)]: Invalid argument"; break;
        case ENFILE: errorMessage = "[ENFILE (23)]: File table overflow"; break;
        case EMFILE: errorMessage = "[EMFILE (24)]: Too many open files"; break;
        case ENOTTY: errorMessage = "[ENOTTY (25)]: Not a typewriter"; break;
        case ETXTBSY: errorMessage = "[ETXTBSY (26)]: Text file busy"; break;
        case EFBIG: errorMessage = "[EFBIG (27)]: File too large"; break;
        case ENOSPC: errorMessage = "[ENOSPC (28)]: No space left on device"; break;
        case ESPIPE: errorMessage = "[ESPIPE (29)]: Illegal seek"; break;
        case EROFS: errorMessage = "[EROFS (30)]: Read-only file system"; break;
        case EMLINK: errorMessage = "[EMLINK (31)]: Too many links"; break;
        case EPIPE: errorMessage = "[EPIPE (32)]: Broken pipe"; break;
        case EDOM: errorMessage = "[EDOM (33)]: Math argument out of domain of func"; break;
        case ERANGE: errorMessage = "[ERANGE (34)]: Math result not representable"; break;
            
        case EDEADLK: errorMessage = "[EDEADLK (35)]: Resource deadlock would occur"; break;
        case ENAMETOOLONG: errorMessage = "[ENAMETOOLONG (36)]: File name too long"; break;
        case ENOLCK: errorMessage = "[ENOLCK (37)]: No record locks available"; break;
        case ENOSYS: errorMessage = "[ENOSYS (38)]: Invalid system call number"; break;
        case ENOTEMPTY: errorMessage = "[ENOTEMPTY (39)]: Directory not empty"; break;
        case ELOOP: errorMessage = "[ELOOP (40)]: Too many symbolic links encountered"; break;
        //case EWOULDBLOCK: errorMessage = "[EWOULDBLOCK/EAGAIN (41)]: Operation would block"; break;
        case ENOMSG: errorMessage = "[ENOMSG (42)]: No message of desired type"; break;
        case EIDRM: errorMessage = "[EIDRM (43)]: Identifier removed"; break;
        case ECHRNG: errorMessage = "[ECHRNG (44)]: Channel number out of range"; break;
        case EL2NSYNC: errorMessage = "[EL2NSYNC (45)]: Level 2 not synchronized"; break;
        case EL3HLT: errorMessage = "[EL3HLT (46)]: Level 3 halted"; break;
        case EL3RST: errorMessage = "[EL3RST (47)]: Level 3 reset"; break;
        case ELNRNG: errorMessage = "[ELNRNG (48)]: Link number out of range"; break;
        case EUNATCH: errorMessage = "[EUNATCH (49)]: Protocol driver not attached"; break;
        case ENOCSI: errorMessage = "[ENOCSI (50)]: No CSI structure available"; break;
        case EL2HLT: errorMessage = "[EL2HLT (51)]: Level 2 halted"; break;
        case EBADE: errorMessage = "[EBADE (52)]: Invalid exchange"; break;
        case EBADR: errorMessage = "[EBADR (53)]: Invalid request descriptor"; break;
        case EXFULL: errorMessage = "[EXFULL (54)]: Exchange full"; break;
        case ENOANO: errorMessage = "[ENOANO (55)]: No anode"; break;
        case EBADRQC: errorMessage = "[EBADRQC (56)]: Invalid request code"; break;
        case EBADSLT: errorMessage = "[EBADSLT (57)]: Invalid slot"; break;
        //case EDEADLOCK: errorMessage = "[EDEADLOCK (35)]: Resource deadlock would occur"; break;
        case EBFONT: errorMessage = "[EBFONT (59)]: Bad font file format"; break;
        case ENOSTR: errorMessage = "[ENOSTR (60)]: Device not a stream"; break;
        case ENODATA: errorMessage = "[ENODATA (61)]: No data available"; break;
        case ETIME: errorMessage = "[ETIME (62)]: Timer expired"; break;
        case ENOSR: errorMessage = "[ENOSR (63)]: Out of streams resources"; break;
        case ENONET: errorMessage = "[ENONET (64)]: Machine is not on the network"; break;
        case ENOPKG: errorMessage = "[ENOPKG (65)]: Package not installed"; break;
        case EREMOTE: errorMessage = "[EREMOTE (66)]: Object is remote"; break;
        case ENOLINK: errorMessage = "[ENOLINK (67)]: Link has been severed"; break;
        case EADV: errorMessage = "[EADV (68)]: Advertise error"; break;
        case ESRMNT: errorMessage = "[ESRMNT (69)]: Srmount error"; break;
        case ECOMM: errorMessage = "[ECOMM (70)]: Communication error on send"; break;
        case EPROTO: errorMessage = "[EPROTO (71)]: Protocol error"; break;
        case EMULTIHOP: errorMessage = "[EMULTIHOP (72)]: Multihop attempted"; break;
        case EDOTDOT: errorMessage = "[EDOTDOT (73)]: RFS specific error"; break;
        case EBADMSG: errorMessage = "[EBADMSG (74)]: Not a data message"; break;
        case EOVERFLOW: errorMessage = "[EOVERFLOW (75)]: Value too large for defined data type"; break;
        case ENOTUNIQ: errorMessage = "[ENOTUNIQ (76)]: Name not unique on network"; break;
        case EBADFD: errorMessage = "[EBADFD (77)]: File descriptor in bad state"; break;
        case EREMCHG: errorMessage = "[EREMCHG (78)]: Remote address changed"; break;
        case ELIBACC: errorMessage = "[ELIBACC (79)]: Can not access a needed shared library"; break;
        case ELIBBAD: errorMessage = "[ELIBBAD (80)]: Accessing a corrupted shared library"; break;
        case ELIBSCN: errorMessage = "[ELIBSCN (81)]: .lib section in a.out corrupted"; break;
        case ELIBMAX: errorMessage = "[ELIBMAX (82)]: Attempting to link in too many shared libraries"; break;
        case ELIBEXEC: errorMessage = "[ELIBEXEC (83)]: Cannot exec a shared library directly"; break;
        case EILSEQ: errorMessage = "[EILSEQ (84)]: Illegal byte sequence"; break;
        case ERESTART: errorMessage = "[ERESTART (85)]: Interrupted system call should be restarted"; break;
        case ESTRPIPE: errorMessage = "[ESTRPIPE (86)]: Streams pipe error"; break;
        case EUSERS: errorMessage = "[EUSERS (87)]: Too many users"; break;
        case ENOTSOCK: errorMessage = "[ENOTSOCK (88)]: Socket operation on non-socket"; break;
        case EDESTADDRREQ: errorMessage = "[EDESTADDRREQ (89)]: Destination address required"; break;
        case EMSGSIZE: errorMessage = "[EMSGSIZE (90)]: Message too long"; break;
        case EPROTOTYPE: errorMessage = "[EPROTOTYPE (91)]: Protocol wrong type for socket"; break;
        case ENOPROTOOPT: errorMessage = "[ENOPROTOOPT (92)]: Protocol not available"; break;
        case EPROTONOSUPPORT: errorMessage = "[EPROTONOSUPPORT (93)]: Protocol not supported"; break;
        case ESOCKTNOSUPPORT: errorMessage = "[ESOCKTNOSUPPORT (94)]: Socket type not supported"; break;
        case EOPNOTSUPP: errorMessage = "[EOPNOTSUPP (95)]: Operation not supported on transport endpoint"; break;
        case EPFNOSUPPORT: errorMessage = "[EPFNOSUPPORT (96)]: Protocol family not supported"; break;
        case EAFNOSUPPORT: errorMessage = "[EAFNOSUPPORT (97)]: Address family not supported by protocol"; break;
        case EADDRINUSE: errorMessage = "[EADDRINUSE (98)]: Address already in use"; break;
        case EADDRNOTAVAIL: errorMessage = "[EADDRNOTAVAIL (99)]: Cannot assign requested address"; break;
        case ENETDOWN: errorMessage = "[ENETDOWN (100)]: Network is down"; break;
        case ENETUNREACH: errorMessage = "[ENETUNREACH (101)]: Network is unreachable"; break;
        case ENETRESET: errorMessage = "[ENETRESET (102)]: Network dropped connection because of reset"; break;
        case ECONNABORTED: errorMessage = "[ECONNABORTED (103)]: Software caused connection abort"; break;
        case ECONNRESET: errorMessage = "[ECONNRESET (104)]: Connection reset by peer"; break;
        case ENOBUFS: errorMessage = "[ENOBUFS (105)]: No buffer space available"; break;
        case EISCONN: errorMessage = "[EISCONN (106)]: Transport endpoint is already connected"; break;
        case ENOTCONN: errorMessage = "[ENOTCONN (107)]: Transport endpoint is not connected"; break;
        case ESHUTDOWN: errorMessage = "[ESHUTDOWN (108)]: Cannot send after transport endpoint shutdown"; break;
        case ETOOMANYREFS: errorMessage = "[ETOOMANYREFS (109)]: Too many references: cannot splice"; break;
        case ETIMEDOUT: errorMessage = "[ETIMEDOUT (110)]: Connection timed out"; break;
        case ECONNREFUSED: errorMessage = "[ECONNREFUSED (111)]: Connection refused"; break;
        case EHOSTDOWN: errorMessage = "[EHOSTDOWN (112)]: Host is down"; break;
        case EHOSTUNREACH: errorMessage = "[EHOSTUNREACH (113)]: No route to host"; break;
        case EALREADY: errorMessage = "[EALREADY (114)]: Operation already in progress"; break;
        case EINPROGRESS: errorMessage = "[EINPROGRESS (115)]: Operation now in progress"; break;
        case ESTALE: errorMessage = "[ESTALE (116)]: Stale file handle"; break;
        case EUCLEAN: errorMessage = "[EUCLEAN (117)]: Structure needs cleaning"; break;
        case ENOTNAM: errorMessage = "[ENOTNAM (118)]: Not a XENIX named type file"; break;
        case ENAVAIL: errorMessage = "[ENAVAIL (119)]: No XENIX semaphores available"; break;
        case EISNAM: errorMessage = "[EISNAM (120)]: Is a named type file"; break;
        case EREMOTEIO: errorMessage = "[EREMOTEIO (121)]: Remote I/O error"; break;
        case EDQUOT: errorMessage = "[EDQUOT (122)]: Quota exceeded"; break;
        case ENOMEDIUM: errorMessage = "[ENOMEDIUM (123)]: No medium found"; break;
        case EMEDIUMTYPE: errorMessage = "[EMEDIUMTYPE (124)]: Wrong medium type"; break;
        case ECANCELED: errorMessage = "[ECANCELED (125)]: Operation Canceled"; break;
        case ENOKEY: errorMessage = "[ENOKEY (126)]: Required key not available"; break;
        case EKEYEXPIRED: errorMessage = "[EKEYEXPIRED (127)]: Key has expired"; break;
        case EKEYREVOKED: errorMessage = "[EKEYREVOKED (128)]: Key has been revoked"; break;
        case EKEYREJECTED: errorMessage = "[EKEYREJECTED (129)]: Key was rejected by service"; break;
        case EOWNERDEAD: errorMessage = "[EOWNERDEAD (130)]: Owner died"; break;
        case ENOTRECOVERABLE: errorMessage = "[ENOTRECOVERABLE (131)]: State not recoverable"; break;
        case ERFKILL: errorMessage = "[ERFKILL (132)]: Operation not possible due to RF-kill"; break;
        case EHWPOISON: errorMessage = "[EHWPOISON (133)]: Memory page has hardware error"; break;
        default: errorMessage = "Unknown error code: " + std::to_string(errno);
        }
    // clang-format on

    return errorMessage;
}