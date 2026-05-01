/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class', // <--- FUNDAMENTAL para que funcione el botón
  content: [
    './src/**/*.{html,ts}',
  ],
  theme: {
    extend: {
      animation: {
        'bounce-slow': 'bounce 3s infinite',
      }
    },
  },
  plugins: [],
}