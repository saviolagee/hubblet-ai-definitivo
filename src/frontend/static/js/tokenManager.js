const TOKEN_STORAGE_KEY = 'hubbletUserTokenUsage';
const DEFAULT_TOTAL_TOKENS = 2000000;

/**
 * @typedef {object} TokenUsage
 * @property {number} totalTokens - Total de tokens disponíveis para o usuário.
 * @property {number} usedTokens - Total de tokens já utilizados pelo usuário.
 */

/**
 * Obtém os dados de uso de tokens do localStorage.
 * Se não houver dados, inicializa com os valores padrão.
 * @returns {TokenUsage}
 */
function getTokenUsage() {
  const storedUsage = localStorage.getItem(TOKEN_STORAGE_KEY);
  if (storedUsage) {
    return JSON.parse(storedUsage);
  }
  const initialUsage = {
    totalTokens: DEFAULT_TOTAL_TOKENS,
    usedTokens: 0,
  };
  localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(initialUsage));
  return initialUsage;
}

/**
 * Salva os dados de uso de tokens no localStorage.
 * @param {TokenUsage} usage - O objeto TokenUsage para salvar.
 */
function setTokenUsage(usage) {
  localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(usage));
  // Disparar um evento customizado para notificar a UI sobre a mudança
  window.dispatchEvent(new CustomEvent('tokenUsageChanged', { detail: usage }));
}

/**
 * Estima o número de tokens em um texto (caracteres / 4).
 * @param {string} text - O texto para calcular os tokens.
 * @returns {number} - O número estimado de tokens.
 */
function countTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

/**
 * Atualiza os tokens usados após uma interação no chat.
 * @param {string} inputText - O texto de entrada do usuário.
 * @param {string} outputText - O texto de saída do modelo.
 * @returns {TokenUsage} - O estado atualizado do uso de tokens.
 */
function updateUsedTokens(inputText, outputText) {
  const usage = getTokenUsage();
  const inputTokens = countTokens(inputText);
  const outputTokens = countTokens(outputText);
  usage.usedTokens += inputTokens + outputTokens;
  setTokenUsage(usage);
  console.log(`Tokens updated: Input=${inputTokens}, Output=${outputTokens}, Used=${usage.usedTokens}, Total=${usage.totalTokens}`);
  return usage;
}

/**
 * Adiciona 1 milhão de tokens ao total de tokens do usuário.
 * @returns {TokenUsage} - O estado atualizado do uso de tokens.
 */
function addMillionTokens() {
  const usage = getTokenUsage();
  usage.totalTokens += 1000000;
  setTokenUsage(usage);
  console.log(`Added 1M tokens. New total: ${usage.totalTokens}`);
  return usage;
}

/**
 * Verifica se o usuário atingiu o limite de tokens.
 * @returns {boolean}
 */
function hasReachedLimit() {
  const usage = getTokenUsage();
  const limitReached = usage.usedTokens >= usage.totalTokens;
  if (limitReached) {
    console.warn('Token limit reached!');
  }
  return limitReached;
}

// Para facilitar a integração, vamos expor as funções globalmente 
// ou você pode adaptar para seu sistema de módulos (AMD, CommonJS, ES6 Modules)
window.hubbletTokenManager = {
  getTokenUsage,
  setTokenUsage,
  countTokens,
  updateUsedTokens,
  addMillionTokens,
  hasReachedLimit
};

// Inicializa e dispara o evento para a UI carregar os valores iniciais
document.addEventListener('DOMContentLoaded', () => {
  const initialUsage = getTokenUsage();
  window.dispatchEvent(new CustomEvent('tokenUsageChanged', { detail: initialUsage }));
});
