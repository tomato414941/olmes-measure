"""What the language-model projects share of their evaluations: the tools that measure a published benchmark as its
publisher defines it and that find the text that overlaps it, and the evaluation definitions that projects adopt.

- olmes_measure (beside this package): AI2's OLMES suites, from the requests that the suites' scores read.
- paloma: AI2's Paloma, its text read as Paloma reads it.
- overlap: the documents that hold a benchmark's questions or text, which nothing that is measured by it may learn.
- definition: evaluation definitions by name (definitions/NAME.json): the data that is measured, how it is divided
  into categories and how two results compare. A definition is not edited once it is used: a change is a new name.

The tools know of no definition. A project chooses the definition that it measures by, and how it reads its own
training text; a definition that is `within` another measures a part of the other's data, so that a result by the
larger one can be compared with it on the part.
"""
