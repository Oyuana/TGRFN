/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        risk: { high: '#ef4444', mid: '#f59e0b', safe: '#22c55e' },
      },
    },
  },
  plugins: [],
}
