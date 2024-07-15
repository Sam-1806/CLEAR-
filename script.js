document.querySelector(".hamb").addEventListener("click", () => {
    console.log("hi");
    document.querySelector(".right-sub").style.left = "20px";
    document.querySelector(".form").style.filter = "opacity(20%)";
   
    document.querySelector(".bar").style.filter = "opacity(20%)";
    
    
  });
document.querySelector(".cross").addEventListener("click",()=>{
    console.log("hi");
    document.querySelector(".right-sub").style.left="-150px";
    document.querySelector(".form").style.filter = "opacity(100%)";
    
    

})
const prevButton = document.querySelector('.prev');
const nextButton = document.querySelector('.next');
const testimonialCards = document.querySelector('.testimonial-cards');
let currentIndex = 0;

prevButton.addEventListener('click', () => {
  if (currentIndex > 0) {
    currentIndex--;
    updateCards();
  }
});

nextButton.addEventListener('click', () => {
  if (currentIndex < testimonialCards.children.length - 1) {
    currentIndex++;
    updateCards();
  }
});

function updateCards() {
  const offset = -currentIndex * 100;
  testimonialCards.style.transform = `translateX(${offset}%)`;
}
