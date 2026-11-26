export default [{
  files: ["**/*.mjs"],
  languageOptions: {
    ecmaVersion: 2022,
    sourceType: "module",
    globals: { console: "readonly", process: "readonly", structuredClone: "readonly", URL: "readonly" },
  },
  rules: {
    "no-undef": "error",
    "no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
    "no-unreachable": "error",
    "no-constant-condition": "error",
    "valid-typeof": "error",
    "constructor-super": "error",
  },
}];
