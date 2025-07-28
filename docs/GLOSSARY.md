# Glossary of Terms

This document provides definitions for key terms used throughout the KineticAugment documentation.

---

### A

**Anatomical Plausibility**
: The principle that a pose or motion must be achievable by a real human body, respecting constraints like joint angle limits and bone lengths.

**Augmentation Priority System**
: A system for categorizing skeletal features (e.g., joints, limbs) into High, Medium, or Low priority tiers to guide how aggressively they can be augmented without destroying the semantic meaning of the motion.

### B

**Bilateral Symmetry**
: The property of being divisible into symmetrical halves. In human motion, this refers to the coordinated relationship between the left and right sides of the body. See also: *Inter-Limb Coordination*.

**Biomechanics**
: The study of the mechanical laws relating to the movement or structure of living organisms.

### C

**Chiral / Chirality**
: A property of an object (or motion) that is not superimposable on its mirror image. For example, writing with one's right hand is a chiral action; a mirrored version would show a person writing with their left hand, which may not be semantically equivalent in all contexts.

### D

**Dynamic Time Warping (DTW)**
: An algorithm for measuring the similarity between two temporal sequences that may vary in speed. In KineticAugment, it is used in the *Validation Framework* to ensure that an augmented movement path has not deviated too much from the original's shape.

### F

**Forward Kinematics**
: A method for calculating the position of the end of a linked structure (like an arm) given the angles of all the joints. It is the primary mechanism for enforcing *Kinematic Chain Consistency* when a parent joint (e.g., the shoulder) is rotated.

**Fréchet Distance**
: A measure of similarity between two curves that takes into account the location and ordering of points along the curves. It's often intuitively described as the "dog-walker's distance." Like DTW, it can be used to validate trajectory similarity.

### G

**Gimbal Lock**
: A problem that occurs when using Euler angles for 3D rotation, where two of the three rotation axes align, causing a loss of one degree of rotational freedom. This can corrupt rotational data and interpolation. It is avoided by using *Quaternions*.

### I

**Inter-Limb Coordination**
: The principle that the movements of different limbs are often correlated (moving together) or anti-correlated (moving in opposition). This is a more general and accurate concept than simple bilateral symmetry.

### K

**Kinematic Chain**
: A sequence of rigid bodies (bones) connected by joints. The human arm (`shoulder -> elbow -> wrist`) is a classic example. The principle of *Kinematic Chain Consistency* ensures these chains are not broken during augmentation.

### N

**Non-Manual Markers (NMMs)**
: In Sign Language Linguistics, these are grammatical or affective signals conveyed by parts of the body other than the hands, such as facial expressions (eyebrow raises, mouth morphemes) and head or body postures. Preserving these is critical for the *Semantic Integrity* of sign language data.

### P

**Perlin / Simplex Noise**
: A type of gradient noise function used to produce natural-looking, smooth, pseudo-random patterns. In KineticAugment, it is ideal for creating `Dynamic Trajectory Jittering` that mimics natural human tremor or path variation without being jerky.

### Q

**Quaternion**
: A four-dimensional number system used to represent 3D rotations. Quaternions are used within KineticAugment for rotation augmentations because they avoid *Gimbal Lock* and allow for smooth and efficient interpolation between orientations.

### S

**Sign Space**
: The three-dimensional area in front of a signer's body where most signs are produced. The location of a sign within this space is often a phonologically meaningful feature that must be preserved within a certain tolerance.

### T

**Task-Specific Profile**
: A configuration file (e.g., `slr_profile.yaml`) that defines a complete augmentation strategy for a specific task. It specifies which augmentations to use, their parameters, and their probabilities, effectively implementing the *Augmentation Priority System*.

### Y

**YAML (YAML Ain't Markup Language)**
: A human-readable data serialization standard. It is the recommended format for creating *Task-Specific Profiles* in KineticAugment due to its clarity and support for comments.