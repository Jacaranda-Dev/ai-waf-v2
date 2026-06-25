# Cross-site scripting

Cross-site scripting which can also written like XSS, is a type of web security vulnerability that injects malicious scripts into trusted websites. Attackers using XSS can perform some of the following:

* Impersonate users
* Steal login credentials
* Modify pages to show falsified forms and attacted links 
* Trick users into performing performing actions unbeknownst to them

```mermaid
flowchart LR
    A[Attacker] -->|Sends malicious input| B[Website]
    B -->|Shows it to everyone| C[Victim]
    C -->|Browser runs it| D[Malicious code executes]
```