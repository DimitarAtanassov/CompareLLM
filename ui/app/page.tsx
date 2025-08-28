// app/page.tsx

import CompareLLMClient from "./components/CompareLLMClient";


export default function Page() {
  // keeping page.tsx as a tiny Server Component improves SSR & follows Next best practices.
  return <CompareLLMClient />;
}