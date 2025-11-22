# Princeton COS 529 Assignments

This repository contains my solutions to assignments from **Princeton COS 529: Advanced Computer Vision** by Jia Deng.  
They are intended to serve as learning references and to spark new ideas for others working through the course.

---

## Estimated Workload
**HW1**: Scribe a lecture and submit your notes.
**HW2**: Review 1–2 hours; Implementation 3–7 hours.  
  For a concise introduction to probabilistic graphical models, I recommend the **CS228 Notes**:  
- Andrew Lin. *CS 228: Probabilistic Graphical Models — Lecture Notes by Prof. Stefano Ermon*. Winter 2024. [Online notes](https://web.stanford.edu/~lindrew/cs228.pdf)
- Volodymyr Kuleshov and Stefano Ermon (with contributions from students and staff). *CS228 Probabilistic Graphical Models Notes*. Based on Stanford CS228. [Online notes](https://ermongroup.github.io/cs228-notes/) (accessed 2024).
---

**HW3**: Estimated time - Review: 1–2 hours; Implementation: 10–12 hours.

**Key Challenges:**
- **Memory Management**: The primary bottleneck is memory constraints. Avoid storing the complete Jacobian matrix in memory. Instead, implement block-wise computation as outlined in the course notes.
- **Efficient Implementation**: For the Levenberg-Marquardt algorithm, use in-place operations to update the diagonal elements rather than creating identity matrices with `np.eye()`, which consumes unnecessary memory.

**HW4**: Estimated time - Review: 1–2 hours; Writing: 2-3 hours.


## Academic Integrity Notice
These solutions are shared solely for **educational reference**.  
Please do **not** copy or submit them as your own work, as this may violate Princeton’s academic integrity policies.  
Instead, use them to guide your understanding, compare approaches, and inspire your own solutions.
